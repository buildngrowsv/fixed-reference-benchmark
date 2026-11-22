"""Room removal, selectivity and net-atlas bundles for E3 and E5."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from common.atlas_bundle import Bundle
from e3.data import TRAIN_DOMAINS, vocab_strings
from e3.trainer import evaluate, probe_states


def removal(model, val: dict) -> dict:
    """Per-domain loss with each room removed (its modules write nothing).

    Rooms are the walled model's rooms, the routed model's experts, and for dense and
    topographic models four contiguous arcs of attention dimensions and hidden units.
    Selectivity index S = (largest increase - mean of the others) / (largest + mean of the
    others); S = 0.5 is the same as the largest increase being three times the others.
    """
    domains = [d for d in TRAIN_DOMAINS if d in val]
    sub = {d: val[d] for d in domains}
    base = evaluate(model, sub)
    rooms = []
    for r in range(model.c.rooms):
        loss = evaluate(model, sub, remove=r)
        delta = {d: max(loss[d] - base[d], 0.0) for d in domains}
        top = max(domains, key=lambda d: delta[d])
        others = float(np.mean([delta[d] for d in domains if d != top]))
        s = (delta[top] - others) / max(delta[top] + others, 1e-9)
        # A room counts only if removing it costs its top domain at least 1% of that domain's loss.
        meaningful = delta[top] >= 0.01 * base[top]
        rooms.append(dict(room=r, loss=loss, increase=delta, domain=top, selectivity=s, meaningful=bool(meaningful),
                          ratio=delta[top] / max(others, 1e-9)))
    counted = [x for x in rooms if x["meaningful"]]
    covered = sorted({x["domain"] for x in counted if x["selectivity"] >= 0.5})
    return dict(base=base, rooms=rooms, max_selectivity=max([x["selectivity"] for x in counted], default=0.0),
                domains_with_selective_room=covered)


def room_regions(cfg, layers: int) -> list[dict]:
    """Rooms (walled) or arcs (other arms) as channel rectangles in index layout."""
    if cfg.arm == "walled":
        w = cfg.room_width
        regions = [dict(label=f"Room {r + 1}", channels=list(range(r * w, (r + 1) * w)), from_state=0, to_state=layers)
                   for r in range(cfg.rooms)]
        regions.append(dict(label="Doors (shared)", channels=list(range(cfg.rooms * w, cfg.width)),
                            from_state=0, to_state=layers))
        return regions
    q = cfg.width // cfg.rooms
    return [dict(label=f"Arc {r + 1}", channels=list(range(r * q, (r + 1) * q)), from_state=0, to_state=layers)
            for r in range(cfg.rooms)]


def grid(width: int) -> list[int]:
    cols = 32 if width % 32 == 0 and width >= 1024 else 16 if width % 16 == 0 else 8
    if width == 768:
        cols = 32
    return [width // cols, cols]


def bundle(out: Path, *, title: str, description: str, cfg, layers: int, probe_ids: np.ndarray, probe_labels,
           runs: dict, conditions: list[dict], frames: list[int] | None = None, compare: dict | None = None,
           regions: list[dict] | None = None, annotations: list[dict] | None = None, scale_states=None) -> Path:
    """Write a bundle. ``runs`` maps run id -> dict(states=[L+1, items, D], condition, step, label)."""
    b = Bundle(title, description)
    D = cfg.width
    b["model"] = dict(name=f"E3 {cfg.arm} {cfg.d}", width=D, blocks=layers, grid=grid(D))
    b["layer_labels"] = ["Embedding"] + [f"Block {i}" for i in range(1, layers + 1)]
    P, T = probe_ids.shape
    b["items"] = [dict(kind="token", t=t, token=int(probe_ids[p, t]), probe=probe_labels[p])
                  for p in range(P) for t in range(T)]
    b["answer_items"] = [P * T, P * T]
    b["default_item"] = 2 * T - 1 if P > 1 else T - 1
    b["default_layout"] = "index"
    b["vocab"] = vocab_strings(probe_ids)
    b["conditions"] = conditions
    if frames:
        b["frames"] = frames
    if compare:
        b["compare"] = compare
    if regions:
        b["regions"] = regions
    if annotations:
        b["annotations"] = annotations
    b["notes"] = dict(probes=list(probe_labels), probe_tokens=T, config=cfg.to_dict())
    b["runs"] = [dict(id=k, condition=v["condition"], step=v.get("step", 0), parent=None, states_from=0,
                      label=v.get("label", k)) for k, v in runs.items()]
    ref = (next(iter(runs.values()))["states"] if scale_states is None else scale_states).astype(np.float32)
    b.array("layout/channels", np.arange(D, dtype=np.int32), "i32")
    b.array("scale/states", np.sqrt((ref ** 2).mean(1)) + 1e-6, "f32")
    for k, v in runs.items():
        b.array(f"states/{k}", v["states"], "f16")
    return b.save(out)


def training_bundle(out: Path, *, model, cfg, arm: str, label: str, description: str, probe_ids, probe_labels,
                    snapshots: dict, compare: dict | None = None, extra_runs: dict | None = None,
                    annotations: list[dict] | None = None) -> Path:
    """Timelapse bundle of one training run: a trained condition whose frames are the snapshots.

    The first snapshot must be step 0 (the viewer shows it as the condition itself).
    """
    steps = sorted(snapshots)
    assert steps[0] == 0, steps
    runs = {(arm if s == 0 else f"{arm}@{s}"): dict(states=snapshots[s], condition=arm, step=s) for s in steps}
    runs.update(extra_runs or {})
    conditions = [dict(id=arm, label=label, description=description, trained=True)]
    return bundle(out, title=label, description=description, cfg=cfg, layers=len(model.blocks), probe_ids=probe_ids,
                  probe_labels=probe_labels, runs=runs, conditions=conditions, frames=steps, compare=compare,
                  regions=room_regions(cfg, len(model.blocks)), annotations=annotations,
                  scale_states=snapshots[steps[-1]])


def removal_bundle(out: Path, *, model, cfg, title: str, description: str, probe_ids, probe_labels, room: int,
                   room_domain: str) -> Path:
    intact = probe_states(model, probe_ids)
    removed = probe_states(model, probe_ids, remove=room)
    runs = {"intact": dict(states=intact, condition="intact", label="Intact"),
            "removed": dict(states=removed, condition="removed", label=f"Room {room + 1} removed")}
    conditions = [dict(id="intact", label="Intact model", description="All rooms present."),
                  dict(id="removed", label=f"Room {room + 1} removed",
                       description=f"Room {room + 1}'s modules write nothing. It is the room whose removal hurts {room_domain} most.")]
    return bundle(out, title=title, description=description, cfg=cfg, layers=len(model.blocks), probe_ids=probe_ids,
                  probe_labels=probe_labels, runs=runs, conditions=conditions,
                  compare=dict(label="Change vs intact model", ref={"removed": "intact", "intact": "intact"}),
                  regions=room_regions(cfg, len(model.blocks)))
