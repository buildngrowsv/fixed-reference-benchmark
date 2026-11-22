"""Train one E3 arm from scratch (prog-e3-dense, -walled-s1, -walled-s2, -topographic, -routed).

    python run.py e3.arm --arm walled --seed 1 --tokens 1000000000
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from common.job import Job, read_json, write_json
from common.paths import job_output
from e3 import configs, data, measure
from e3.model import Model, count_params, wiring_cost, wiring_distances, wiring_prox
from e3.trainer import save_model, train


def topo_lambda_from_screen() -> float:
    result = read_json(job_output("prog-e3-screen") / "result.json", {}) or {}
    return float(result.get("topo_lambda", configs.TOPO_DEFAULT))


def run_arm(job: Job, *, scale: str, arm: str, seed: int, tokens: int, out: Path, lam: float = 0.0,
            snapshots: int = 6, eval_every: int | None = None, val_windows: int = 64, label: str | None = None,
            bundle_dir: Path | None = None) -> dict:
    """Train, save, measure room removal and write the timelapse bundle. Idempotent."""
    out = Path(out)
    done = read_json(out / "summary.json")
    if done:
        job.log(f"{arm} ({scale}, seed {seed}) already finished")
        return done
    cfg = configs.config(scale, arm, lam)
    mx.random.seed(seed)
    model = Model(cfg)
    mx.eval(model.parameters())
    steps = max(1, tokens // configs.TOKENS_PER_STEP)
    per_domain_tokens = steps * configs.PER_DOMAIN * configs.SEQ + 1
    mix = data.Mixture(per_domain_tokens, configs.SEQ, configs.PER_DOMAIN, seed)
    epochs = {d: round(steps * configs.PER_DOMAIN / w.count, 3) for d, w in zip(mix.domains, mix.windows)}
    val = data.validation(configs.SEQ, val_windows)
    probe_ids, probe_labels = data.probes(24)
    fpt = configs.flops_per_token(cfg, cfg.layers)
    params = count_params(model)
    job.log(f"{arm}: {params} parameters, {steps} steps, {steps * configs.TOKENS_PER_STEP:,} tokens, epochs {epochs}")

    after = on_eval = None
    if arm == "topographic":
        dist = wiring_distances(cfg)
        after = lambda m, lr: wiring_prox(m, dist, lam, lr)  # noqa: E731
        on_eval = lambda m: dict(wiring_cost=wiring_cost(m, dist))  # noqa: E731
    t0 = time.time()
    res = train(job, model, arm=f"{arm}-{scale}-s{seed}", steps=steps, peak_lr=configs.PEAK_LR[scale],
                batches=mix.get, val=val, out=out, tokens_per_step=configs.TOKENS_PER_STEP,
                eval_every=eval_every or max(50, steps // 20), snapshots=snapshots, probe_ids=probe_ids,
                after_update=after, flops_per_token=fpt, on_eval=on_eval, micro=configs.MICRO[scale])
    hours = (time.time() - t0) / 3600
    save_model(model, cfg, out / "model", dict(seed=seed, steps=steps, tokens=steps * configs.TOKENS_PER_STEP))
    rem = measure.removal(model, val)
    final = res["history"][-1]
    label = label or f"{arm.capitalize()} {scale} model, seed {seed}"
    descriptions = {
        "dense": "Every module reads and writes the whole residual stream.",
        "walled": f"Four rooms of {cfg.room_width} channels and {cfg.doors} shared door channels. Each room's "
                  "attention and feed-forward read and write only their room and the doors.",
        "topographic": f"Dense model whose units sit on a circle; long connections are penalized (lambda {lam}).",
        "routed": "Dense attention; four feed-forward room experts with a learned top-2 router.",
    }
    measure.training_bundle(bundle_dir or (out / "atlas"), model=model, cfg=cfg, arm=arm, label=label,
                            description=descriptions[arm] + " Trained from scratch on English, Python and Welsh. "
                            "Frames are training snapshots; probes are one text per domain.",
                            probe_ids=probe_ids, probe_labels=probe_labels, snapshots=res["snapshots"])
    summary = dict(arm=arm, scale=scale, seed=seed, lam=lam, steps=steps, tokens=steps * configs.TOKENS_PER_STEP,
                   params=params, flops_per_token=fpt, train_hours=hours,
                   tokens_per_second=steps * configs.TOKENS_PER_STEP / max(hours * 3600, 1e-6), data_epochs=epochs,
                   val={k[4:]: v for k, v in final.items() if k.startswith("val_")},
                   train_mean_val=float(np.mean([final[f"val_{d}"] for d in data.TRAIN_DOMAINS])),
                   removal=rem, wiring_cost=final.get("wiring_cost"), peak_memory_gb=mx.get_peak_memory() / 1e9)
    write_json(out / "summary.json", summary)
    shutil.rmtree(out / "ckpt", ignore_errors=True)  # resume state; the trained model is saved above
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["dense", "walled", "topographic", "routed"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=400_000_000)
    ap.add_argument("--lam", type=float, default=None, help="topographic penalty; default from prog-e3-screen")
    ap.add_argument("--scale", default="main", choices=["main", "screen"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.tokens = 40 * configs.TOKENS_PER_STEP
    lam = (a.lam if a.lam is not None else topo_lambda_from_screen()) if a.arm == "topographic" else 0.0
    job = Job(f"prog-e3-{a.arm}-s{a.seed}" + ("-smoke" if a.smoke else ""), dict(vars(a), lam=lam))
    s = run_arm(job, scale=a.scale, arm=a.arm, seed=a.seed, tokens=a.tokens, out=job.dir, lam=lam,
                eval_every=10 if a.smoke else None, val_windows=8 if a.smoke else 64)
    job.finish("trained", **s)


if __name__ == "__main__":
    main()
