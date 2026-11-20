"""Separate the current boundary-jump response from the accumulated saved words."""
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

import mlx.core as mx
import numpy as np

from anchored_witness_routing import routing
from inferred_regions import evaluate
from long_hold import HERE, SEEDS
from probe_scale import SIDE
from spatial_witness import recorded_step
from witness_staggered_jumps import WORLDS, schedule

PRIOR = "spatial-witness-staggered-jumps-7077888"
REVISION = "2782d08b0793dc280241fdcf3dc123528de82ff0"


def summarize(original, common, plans):
    archived = np.stack([original[i, sorted(plan), :, 1] for i, plan in enumerate(plans)])
    reset = np.stack([common[i, sorted(plan), :, 1] for i, plan in enumerate(plans)])
    events = np.stack([archived, reset])
    counts = events.sum(axis=1)
    by_jump = counts[..., 3] / counts[..., 2]
    pooled = events.sum(axis=(1, 2))
    rate = pooled[..., 3] / pooled[..., 2]
    wins = rate[:, 1] - rate[:, 0]
    draws = np.random.default_rng(708919).integers(len(plans), size=(5000, len(plans)))
    tally = events.sum(axis=2)[:, draws].sum(axis=2)
    sampled_rate = tally[..., 3] / tally[..., 2]
    gains = sampled_rate[..., 1] - sampled_rate[..., 0]
    effect = wins[0] - wins[1]
    bounds = np.percentile(gains[0] - gains[1], [2.5, 97.5])
    decision = "unresolved"
    if effect >= .01 and bounds[0] > 0:
        decision = "retention_history_widens_gain"
    elif effect <= -.01 and bounds[1] < 0:
        decision = "retention_history_narrows_gain"
    elif bounds[0] >= -.01 and bounds[1] <= .01:
        decision = "gain_robust_to_common_start"
    return dict(decision=decision, retention=rate.tolist(), by_jump=by_jump.tolist(),
                observed_sampled_minus_stale=wins.tolist(),
                gain_change_from_history=dict(mean=float(effect), interval=bounds.tolist()))


def check(original, common, earlier_offsets, offsets):
    np.testing.assert_array_equal(common[:, :, 2], original[:, :, 2])
    np.testing.assert_array_equal(offsets, earlier_offsets)
    # The memory intervention must preserve write offers, acceptance and false writes.
    fields = [0, 2, 4, 5]
    np.testing.assert_array_equal(common[..., fields], original[..., fields])


def main():
    start = time.perf_counter()
    mx.set_cache_limit(2 * 1024**3)
    seed = SEEDS[0]
    payload = (HERE / "results" / f"cube-full-{seed}.json").read_bytes()
    driver = recorded_step(json.loads(payload))
    if sys.argv[1:] == ["--smoke"]:
        plan = {2: np.array([4, 0, 0])}
        params = dict(world=48, side=16, rounds=4, stratified=True, sampled_routes=routing(48, 16),
                      boundary_steps=plan)
        old, offsets = evaluate(driver, **params)
        new, new_offsets = evaluate(driver, **params, reset_saved_at_boundary_steps=True)
        check(old[None], new[None], offsets, new_offsets)
        print("16³ paired common-start smoke verified unchanged writes and supplied controls")
        return
    root = Path(os.environ["STUDIO_QUEUE_ROOT"])
    out = Path(os.environ["STUDIO_JOB_DIR"])
    prior = root / "runs" / PRIOR
    receipt = json.loads((prior / "receipt.json").read_text())
    assert (receipt["revision"], receipt["status"], receipt["exit"]) == (REVISION, "done", 0)
    report = json.loads((prior / "result.json").read_text())
    assert report["decision"] == "unresolved" and sha256(payload).hexdigest() == report["checkpoint_digest"]
    worlds = list(WORLDS)
    assert worlds == report["worlds"]
    plans = [schedule(w) for w in worlds]
    assert json.loads(json.dumps([{k:v.tolist() for k,v in plan.items()} for plan in plans])) == report["schedules"]
    source = (prior / "staggered.npz").read_bytes()
    with np.load(prior / "staggered.npz") as raw:
        old, earlier_offsets = raw["counts"], raw["shifts"]
    new = []
    for w, plan in zip(worlds, plans, strict=True):
        new.append(evaluate(driver, w, side=SIDE, stratified=True, sampled_routes=routing(w, SIDE),
                            boundary_steps=plan, reset_saved_at_boundary_steps=True))
        print(f"Completed common-start world {w}", flush=True)
    counts, offsets = (np.stack(v) for v in zip(*new, strict=True))
    check(old, counts, earlier_offsets, offsets)
    evidence = summarize(old, counts, plans)
    np.testing.assert_allclose(evidence["retention"][0], report["first_round"]["retention"])
    np.savez_compressed(out / "common-start.npz", counts=counts, shifts=offsets)
    evidence.update(worlds=worlds, cells=SIDE**3, baseline_sha256=sha256(source).hexdigest(),
                    checkpoint_digest=sha256(payload).hexdigest(), runtime_s=time.perf_counter()-start,
                    peak_mlx_bytes=mx.get_peak_memory())
    (out / "result.json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
