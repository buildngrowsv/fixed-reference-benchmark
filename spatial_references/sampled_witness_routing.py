"""Use the same sampled detector lines to choose boundaries and witness axes."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import mlx.core as mx
import numpy as np

from inferred_regions import PROBES, ROUNDS, SIDE, WORLDS, compare_sparse, evaluate, summarize
from long_hold import HERE, SEEDS
from probe_scale import SIDE as LARGE_SIDE, WORLD_IDS as LARGE_WORLDS
from spatial_witness import recorded_step

PRIOR = "spatial-stratified-region-probes"
PRIOR_REVISION = "24739b3275db87b8e1d05a0a9b7cc861381f0e73"
LARGE_PRIOR = "spatial-fixed-probe-count-7077888"
LARGE_REVISION = "a3392ce86577280a96aa94f6aad5e9519576c60b"


def choose_axes(active, shift, probes, distance):
    side = active.shape[0]
    half = side // 2
    rates = np.empty((8, 3))
    axial = np.arange(side)
    for axis, pixels in enumerate(probes):
        order = list(range(3))
        order[0], order[axis] = order[axis], order[0]
        axes = np.asarray(order)
        c, d = np.unravel_index(pixels, (side, side))
        # Region number follows row-major x/y/z half membership in divide().
        weights = 2 ** (2 - axes)
        relative = (axial - shift[axis]) % side
        primary = relative // half
        other = ((np.vstack((c, d)) - shift[axes[1:], None]) % side) // half
        regions = primary[:, None] * weights[0] + weights[1:] @ other
        partner = (primary * half + (relative % half - distance) % half + shift[axis]) % side
        lines = active.swapaxes(0, axis).reshape(side, -1)[:, pixels]
        weights_in_region = np.bincount(regions.ravel(), minlength=8)
        joint = np.bincount(regions.ravel(), (lines & lines[partner]).ravel(), minlength=8)
        if np.any(weights_in_region == 0):
            raise ValueError("detector lines did not cover every inferred region")
        rates[:, axis] = joint / weights_in_region
    return rates.argmin(axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--large", action="store_true")
    large = parser.parse_args().large
    started = time.perf_counter()
    mx.set_cache_limit(2 * 1024**3)
    out = Path(os.environ["STUDIO_JOB_DIR"])
    side, worlds = (LARGE_SIDE, LARGE_WORLDS) if large else (SIDE, range(WORLDS))
    job, revision, archive = ((LARGE_PRIOR, LARGE_REVISION, "stratified.npz") if large
                              else (PRIOR, PRIOR_REVISION, "worlds.npz"))
    root = Path(os.environ["STUDIO_QUEUE_ROOT"]) / "runs" / job
    receipt = json.loads((root / "receipt.json").read_text())
    assert (receipt["revision"], receipt["status"], receipt["exit"]) == (revision, "done", 0)
    prior = json.loads((root / "result.json").read_text())
    all_counts, all_shifts, hashes = [], [], {}
    for seed in SEEDS:
        weights = (HERE / "results" / f"cube-full-{seed}.json").read_bytes()
        hashes[str(seed)] = sha256(weights).hexdigest()
        step = recorded_step(json.loads(weights))
        paired = [evaluate(step, w, side=side, stratified=True, sampled_routes=choose_axes) for w in worlds]
        counts, shifts = (np.stack(x) for x in zip(*paired, strict=True))
        all_counts.append(counts)
        all_shifts.append(shifts)
        print(f"Completed checkpoint {seed}", flush=True)
    counts, shifts = np.stack(all_counts), np.stack(all_shifts)
    assert hashes == prior["frozen_checkpoints"]
    source = root / archive
    with np.load(source) as value:
        np.testing.assert_array_equal(shifts, value["shifts"])
    np.savez_compressed(out / "worlds.npz", counts=counts, shifts=shifts)
    comparison = compare_sparse(source, counts)
    metrics = summarize(counts, shifts, side)
    fixed = metrics["inferred_minus_fixed"]["shifted"]
    legitimate = min(q["legitimate"] for q in metrics["rates"]["inferred"].values())
    passes = (min(q["interval"][0] for q in comparison.values()) >= -.01
              and fixed["mean"] >= .04 and fixed["interval"][0] > 0 and legitimate >= .98)
    metrics.update(decision="same_probes_support_routing" if passes else "unresolved",
                   cells=side**3, worlds=list(worlds), transitions=ROUNDS*16, probes_per_plane=PROBES,
                   sample_fraction_upper=3*PROBES/side**2, paired_minus_full_routing=comparison,
                   frozen_checkpoints=hashes, matched_archive_sha256=sha256(source.read_bytes()).hexdigest(),
                   runtime_s=time.perf_counter() - started, peak_mlx_bytes=mx.get_peak_memory())
    (out / "result.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
