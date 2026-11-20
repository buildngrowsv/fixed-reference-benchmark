"""Test the same absolute probe budget on an eight-times larger periodic grid."""
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import mlx.core as mx
import numpy as np

from inferred_regions import PROBES, compare_sparse, evaluate, summarize
from long_hold import HERE, SEEDS
from spatial_witness import recorded_step


SIDE = 192
WORLD_IDS = range(32, 40)
ROUNDS = 16


def main():
    started = time.perf_counter()
    mx.set_cache_limit(2 * 1024**3)
    out = Path(os.environ["STUDIO_JOB_DIR"])
    checkpoints = {}
    data = {mode: [] for mode in ("full", "stratified")}
    for seed in SEEDS:
        weights = (HERE / "results" / f"cube-full-{seed}.json").read_bytes()
        checkpoints[str(seed)] = sha256(weights).hexdigest()
        compiled = recorded_step(json.loads(weights))
        arrays = {m: [] for m in data}
        for world in WORLD_IDS:
            for mode in arrays:
                arrays[mode].append(evaluate(compiled, world, side=SIDE, rounds=ROUNDS,
                                             stratified=mode == "stratified"))
            print(f"Paired world {world}, checkpoint {seed}", flush=True)
        for mode, rows in arrays.items():
            data[mode].append(tuple(np.stack(x) for x in zip(*rows, strict=True)))
    panels = {}
    for mode, items in data.items():
        counts, shifts = (np.stack(x) for x in zip(*items, strict=True))
        np.savez_compressed(out / f"{mode}.npz", counts=counts, shifts=shifts)
        panels[mode] = counts, shifts
    counts, shifts = panels["stratified"]
    _, full_shifts = panels["full"]
    np.testing.assert_array_equal(shifts[..., 1, :, :], full_shifts[..., 1, :, :])
    scores = {m: summarize(*panel, SIDE) for m, panel in panels.items()}
    comparison = compare_sparse(out / "full.npz", counts)
    control = scores["stratified"]["inferred_minus_fixed"]["shifted"]
    floor = min(s["legitimate"] for s in scores["stratified"]["rates"]["inferred"].values())
    meets = (comparison["shifted"]["interval"][0] >= -.02 and control["mean"] >= .04
             and control["interval"][0] > 0 and floor >= .98)
    report = dict(decision="fixed_probe_count_scales" if meets else "unresolved",
                  side=SIDE, cells=SIDE**3, worlds=list(WORLD_IDS), transitions=ROUNDS * 16,
                  probes_per_plane=PROBES, sample_fraction_upper=3 * PROBES / SIDE**2,
                  frozen_checkpoints=checkpoints, stratified_minus_full=comparison, modes=scores,
                  runtime_s=time.perf_counter() - started, peak_mlx_bytes=mx.get_peak_memory())
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
