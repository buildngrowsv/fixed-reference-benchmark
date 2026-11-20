"""Test dedicated regional detector pairs after locating boundaries with sparse lines."""
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

ANCHORS = 1024
DENSE = ("spatial-stratified-region-probes", "24739b3275db87b8e1d05a0a9b7cc861381f0e73")
SHARED = ("spatial-sampled-witness-routing-884736", "5ef2af27507aa921f79c95a22bef7dcee670ccb0")
LARGE = ("spatial-fixed-probe-count-7077888", "a3392ce86577280a96aa94f6aad5e9519576c60b")


def choose_axes(active, shift, anchors, distance):
    side = active.shape[0]
    half = side // 2
    origins = np.array(list(np.ndindex(2, 2, 2)))[:, None, :] * half + shift
    origin_samples = (origins + anchors) % side
    current = active[tuple(np.moveaxis(origin_samples, -1, 0))]
    rates = []
    for axis in range(3):
        other = origin_samples.copy()
        other[..., axis] = (origins[..., axis] + (anchors[..., axis] - distance) % half) % side
        paired = active[tuple(np.moveaxis(other, -1, 0))]
        rates.append((current & paired).mean(axis=1))
    return np.stack(rates, axis=1).argmin(axis=1)


def routing(world, side):
    rng = np.random.default_rng(918447 + world)
    positions = rng.integers(side//2, size=(8, ANCHORS, 3))

    def estimate(active, shift, probes, distance):
        return choose_axes(active, shift, positions, distance)

    return estimate


def published(root, reference, filename="worlds.npz"):
    name, revision = reference
    run = root / name
    receipt = json.loads((run / "receipt.json").read_text())
    assert (receipt["revision"], receipt["status"], receipt["exit"]) == (revision, "done", 0)
    return run / filename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--large", action="store_true")
    large = parser.parse_args().large
    side, worlds = (LARGE_SIDE, LARGE_WORLDS) if large else (SIDE, range(WORLDS))
    started = time.perf_counter()
    mx.set_cache_limit(2 * 1024**3)
    out = Path(os.environ["STUDIO_JOB_DIR"])
    jobs = Path(os.environ["STUDIO_QUEUE_ROOT"]) / "runs"
    dense = published(jobs, LARGE, "stratified.npz") if large else published(jobs, DENSE)
    shared = None if large else published(jobs, SHARED)
    references = (dense,) if large else (dense, shared)
    data, layouts, checkpoints = [], [], {}
    for seed in SEEDS:
        weights = (HERE / "results" / f"cube-full-{seed}.json").read_bytes()
        checkpoints[str(seed)] = sha256(weights).hexdigest()
        step = recorded_step(json.loads(weights))
        runs = []
        for world in worlds:
            runs.append(evaluate(step, world, side=side, stratified=True, sampled_routes=routing(world, side)))
            print(f"Completed world {world} at checkpoint {seed}", flush=True)
        counts, shifts = (np.stack(x) for x in zip(*runs, strict=True))
        data.append(counts)
        layouts.append(shifts)
        print(f"Completed checkpoint {seed}", flush=True)
    counts, shifts = np.stack(data), np.stack(layouts)
    for path in references:
        control = json.loads((path.parent / "result.json").read_text())
        assert control["frozen_checkpoints"] == checkpoints
        with np.load(path) as older:
            np.testing.assert_array_equal(shifts, older["shifts"])
    np.savez_compressed(out / "worlds.npz", counts=counts, shifts=shifts)
    metrics = summarize(counts, shifts, side)
    to_dense = compare_sparse(dense, counts)
    to_shared = compare_sparse(shared, counts) if shared is not None else None
    gain = metrics["inferred_minus_fixed"]["shifted"]
    legitimate = min(r["legitimate"] for r in metrics["rates"]["inferred"].values())
    yes = (min(r["interval"][0] for r in to_dense.values()) >= -.01 and gain["mean"] >= .04
           and gain["interval"][0] > 0 and legitimate >= .98)
    if to_shared is not None:
        benefit = to_shared["shifted"]
        yes = yes and benefit["mean"] >= .02 and benefit["interval"][0] > 0
    label = "dedicated_probes_scale" if large else "dedicated_probes_restore_routing"
    metrics.update(decision=label if yes else "unresolved",
                   cells=side**3, worlds=list(worlds), transitions=ROUNDS*16, probes_per_plane=PROBES,
                   anchors_per_region=ANCHORS, sample_fraction_upper=3*PROBES/side**2+8*ANCHORS*4/side**3,
                   paired_minus_dense=to_dense, paired_minus_shared=to_shared, frozen_checkpoints=checkpoints,
                   comparator_sha256={p.parent.name: sha256(p.read_bytes()).hexdigest() for p in references},
                   runtime_s=time.perf_counter()-started, peak_mlx_bytes=mx.get_peak_memory())
    (out / "result.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
