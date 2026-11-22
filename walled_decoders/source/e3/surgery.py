"""prog-e3-surgery: remove, transplant and add rooms in the trained 125M-class models.

1. Removal: per-room selectivity for every main arm (measured at the end of each arm job).
2. Transplant: the most selective room of walled seed 1 is replaced by the same room of walled
   seed 2 (its block weights and its embedding, unembedding and norm slices). Only door
   weights are refit. Control: the room removed outright, with the same door refit.
3. Growth: a fifth room is added for Icelandic, which no room has seen. Arms: train the new
   room and all door weights; train the new room alone (sealed); full fine-tuning of the
   whole walled model (the forgetting reference). All on the same Icelandic tokens.
"""
from __future__ import annotations

import argparse
import math
import shutil

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_map_with_path

from common import textdata
from common.job import Job, read_json
from common.paths import job_output
from e3 import configs, data, measure
from e3.model import Config, Model
from e3.trainer import evaluate, load_model, train

ARMS = {"dense": "prog-e3-dense", "walled": "prog-e3-walled-s1", "walled_s2": "prog-e3-walled-s2",
        "topographic": "prog-e3-topographic", "routed": "prog-e3-routed"}


def door_mask(model, extra_room: int | None = None, doors: bool = True):
    """Gradient mask: 1 on door weights (and on every weight of ``extra_room``), 0 elsewhere."""
    c = model.c
    w, R = c.room_width, c.rooms

    def f(path, x):
        name = path.split(".")[-1]
        m = mx.zeros(x.shape)
        if name in ("wqkv", "wgu", "n1", "n2", "wo", "wd") and path.startswith("blocks"):
            if doors:
                if name in ("wqkv", "wgu"):
                    m[:, w:, :] = 1
                elif name in ("n1", "n2"):
                    m[:, w:] = 1
                else:
                    m[:, :, w:] = 1
            if extra_room is not None:
                m[extra_room] = 1
        elif name == "embed":
            if doors:
                m[:, R * w:] = 1
            if extra_room is not None:
                m[:, extra_room * w:(extra_room + 1) * w] = 1
        elif name == "unembed":
            if doors:
                m[R * w:, :] = 1
            if extra_room is not None:
                m[extra_room * w:(extra_room + 1) * w, :] = 1
        elif name == "norm":
            if doors:
                m[R * w:] = 1
            if extra_room is not None:
                m[extra_room * w:(extra_room + 1) * w] = 1
        return m
    return tree_map_with_path(f, model.trainable_parameters())


def clone(host) -> Model:
    """An independent copy (MLX in-place updates rebind arrays, so shared objects would leak edits)."""
    m = Model(host.c, len(host.blocks))
    m.update(tree_map(lambda x: mx.array(x), host.parameters()))
    mx.eval(m.parameters())
    return m


def transplant(host, donor, room: int) -> Model:
    """Copy of ``host`` with room ``room`` (blocks, embedding, unembedding, norm slices) from ``donor``."""
    m = clone(host)
    w = host.c.room_width
    lo, hi = room * w, (room + 1) * w
    for bm, bd in zip(m.blocks, donor.blocks):
        for name in ("n1", "n2", "wqkv", "wo", "wgu", "wd"):
            t = getattr(bm, name)
            t[room] = getattr(bd, name)[room]
            setattr(bm, name, t)
    m.embed[:, lo:hi] = donor.embed[:, lo:hi]
    m.unembed[lo:hi, :] = donor.unembed[lo:hi, :]
    m.norm[lo:hi] = donor.norm[lo:hi]
    mx.eval(m.parameters())
    return m


def zero_room(host, room: int) -> Model:
    """Copy of ``host`` whose room ``room`` writes nothing and contributes nothing to the logits."""
    m = clone(host)
    w = host.c.room_width
    for b in m.blocks:
        for name in ("wo", "wd"):
            t = getattr(b, name)
            t[room] = 0
            setattr(b, name, t)
    m.unembed[room * w:(room + 1) * w, :] = 0
    mx.eval(m.parameters())
    return m


def grow(host) -> Model:
    """Add a fifth room initialised so the model computes exactly the same function."""
    c = host.c
    c5 = Config(**{**c.to_dict(), "rooms": c.rooms + 1})
    m = Model(c5, len(host.blocks))
    mx.eval(m.parameters())
    R, w = c.rooms, c.room_width
    for bh, bn in zip(host.blocks, m.blocks):
        for name in ("n1", "n2", "wqkv", "wgu"):
            new = getattr(bn, name)
            setattr(bn, name, mx.concatenate([getattr(bh, name), new[R:]], axis=0))
        for name in ("wo", "wd"):
            new = getattr(bn, name)
            setattr(bn, name, mx.concatenate([getattr(bh, name), mx.zeros_like(new[R:])], axis=0))
    D, D5 = c.width, c5.width
    zeros = lambda *s: mx.zeros(s)  # noqa: E731
    m.embed = mx.concatenate([host.embed[:, :R * w], zeros(c.vocab, w), host.embed[:, R * w:]], axis=1)
    m.unembed = mx.concatenate([host.unembed[:R * w], zeros(w, c.vocab), host.unembed[R * w:]], axis=0)
    # Zero channels lower the final RMS by sqrt(D/D5); rescale so the output is unchanged.
    gain = math.sqrt(D / D5)
    m.norm = mx.concatenate([host.norm[:R * w] * gain, mx.ones((w,)), host.norm[R * w:] * gain])
    mx.eval(m.parameters())
    return m


def icelandic_batches(steps: int, seed: int):
    w = textdata.Windows(textdata.shard(data.GROWTH_DOMAIN, data.TOK, "train", steps * configs.BATCH * configs.SEQ + 1),
                         configs.SEQ, configs.BATCH, seed=seed)
    return w.get


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit-steps", type=int, default=400)
    ap.add_argument("--growth-steps", type=int, default=500)
    ap.add_argument("--smoke", action="store_true", help="use smoke arm outputs under program-scratch")
    a = ap.parse_args()
    if a.smoke:
        a.refit_steps, a.growth_steps = 6, 6
    job = Job("prog-e3-surgery" + ("-smoke" if a.smoke else ""), vars(a))
    src = {k: (job_output(v) if not a.smoke else job.dir.parent / f"{v}-smoke") for k, v in ARMS.items()}
    if a.smoke:
        src["walled"] = job.dir.parent / "prog-e3-walled-s1-smoke"
        src["walled_s2"] = job.dir.parent / "prog-e3-walled-s2-smoke"
        src["dense"] = job.dir.parent / "prog-e3-dense-s1-smoke"
    summaries = {k: read_json(p / "summary.json") for k, p in src.items()}
    val = data.validation(configs.SEQ, 8 if a.smoke else 64)
    probe_ids, probe_labels = data.probes(24)
    report = dict(removal={k: (s or {}).get("removal") for k, s in summaries.items()})
    dense_loss = summaries["dense"]["train_mean_val"]
    walled_loss = summaries["walled"]["train_mean_val"]
    report["walls_cost"] = walled_loss / dense_loss - 1

    host, cfg, _ = load_model(src["walled"] / "model")
    donor, _, _ = load_model(src["walled_s2"] / "model")
    rooms = summaries["walled"]["removal"]["rooms"]
    room = max(rooms, key=lambda r: (r.get("meaningful", True), r["selectivity"]))
    r, dom = room["room"], room["domain"]
    job.log(f"transplant room {r + 1} (most selective, {dom}, S={room['selectivity']:.3f})")
    measure.removal_bundle(job.dir / "atlas-removal", model=host, cfg=cfg,
                           title="Walled model with one room removed",
                           description=f"Walled 125M-class model. Room {r + 1}'s modules write nothing; that room's "
                           f"removal hurts {dom} most. Probes are one text per training domain.",
                           probe_ids=probe_ids, probe_labels=probe_labels, room=r, room_domain=dom)

    mix = data.Mixture(a.refit_steps * configs.PER_DOMAIN * configs.SEQ + 1, configs.SEQ, configs.PER_DOMAIN, seed=7)
    train_val = {d: val[d] for d in data.TRAIN_DOMAINS}
    tp = {"intact": evaluate(host, train_val)}
    variants = {"removed": zero_room(host, r), "transplant": transplant(host, donor, r)}
    for name, m in variants.items():
        tp[f"{name}_before_refit"] = evaluate(m, train_val)
        train(job, m, arm=f"refit-{name}", steps=a.refit_steps, peak_lr=3e-4, batches=mix.get, val=train_val,
              out=job.dir / f"refit-{name}", tokens_per_step=configs.TOKENS_PER_STEP, weight_decay=0.0,
              eval_every=max(1, a.refit_steps // 4), grad_mask=door_mask(m), micro=configs.MICRO["main"])
        tp[f"{name}_after_refit"] = evaluate(m, train_val)
        job.log(name, {k: round(v, 4) for k, v in tp[f"{name}_after_refit"].items()})
    damage = tp["removed_before_refit"][dom] - tp["intact"][dom]
    tp["room"] = r
    tp["domain"] = dom
    tp["recovery"] = (tp["removed_before_refit"][dom] - tp["transplant_after_refit"][dom]) / max(damage, 1e-9)
    tp["control_recovery"] = (tp["removed_before_refit"][dom] - tp["removed_after_refit"][dom]) / max(damage, 1e-9)
    report["transplant"] = tp

    gr = {}
    batches = icelandic_batches(a.growth_steps, seed=11)
    for name, kind in (("grow_doors", "doors"), ("grow_sealed", "sealed"), ("full_finetune", "full")):
        if kind == "full":
            m = clone(host)
            mask = None
        else:
            m = grow(host)
            mask = door_mask(m, extra_room=host.c.rooms, doors=(kind == "doors"))
        before = evaluate(m, val)
        if kind != "full":
            drift = max(abs(before[d] - evaluate(host, {d: val[d]})[d]) for d in data.TRAIN_DOMAINS)
            gr[f"{name}_initial_drift"] = drift
        train(job, m, arm=name, steps=a.growth_steps, peak_lr=6e-4 if kind != "full" else 3e-4, batches=batches,
              val=val, out=job.dir / name, tokens_per_step=configs.TOKENS_PER_STEP,
              weight_decay=0.0 if mask is not None else 0.1, eval_every=max(1, a.growth_steps // 4), grad_mask=mask,
              micro=configs.MICRO["main"])
        after = evaluate(m, val)
        gr[name] = dict(before=before, after=after, icelandic_gain=before["icelandic"] - after["icelandic"],
                        old_domain_change={d: after[d] - before[d] for d in data.TRAIN_DOMAINS},
                        trainable=int(sum(float(x.sum().item()) for _, x in tree_flatten(mask))) if mask else None)
        job.log(name, "icelandic gain", round(gr[name]["icelandic_gain"], 4), "old change",
                {d: round(v, 4) for d, v in gr[name]["old_domain_change"].items()})
    report["growth"] = gr

    walled = summaries["walled"]["removal"]
    selective_all = set(walled["domains_with_selective_room"]) >= set(data.TRAIN_DOMAINS)
    transplant_ok = tp["recovery"] >= 0.9 and tp["recovery"] - tp["control_recovery"] >= 0.25
    cost_ok = report["walls_cost"] <= 0.03
    if selective_all and transplant_ok and cost_ok:
        decision = "walls_success"
    elif walled["max_selectivity"] >= 0.5 or transplant_ok:
        decision = "walls_partial"
    else:
        decision = "walls_null"
    report["criteria"] = dict(selective_room_per_domain=selective_all, transplant_works=transplant_ok,
                              cost_within_3_percent=cost_ok)
    for d in ("refit-removed", "refit-transplant", "grow_doors", "grow_sealed", "full_finetune"):
        shutil.rmtree(job.dir / d / "ckpt", ignore_errors=True)  # resume state only
    job.finish(decision, **report)


if __name__ == "__main__":
    main()
