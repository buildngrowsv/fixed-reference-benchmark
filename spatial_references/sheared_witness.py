"""Infer a discrete shear from the same sparse lines used to locate partition walls."""
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

import mlx.core as mx
import numpy as np

from anchored_witness_routing import ANCHORS, routing
from inferred_regions import PROBES, distributed_positions, evaluate, infer_with_probes, summarize, witness_field
from long_hold import HERE, SEEDS
from probe_scale import SIDE
from spatial_witness import recorded_step

CANDIDATES = (0, -1, 1)
WORLDS = tuple(range(64, 72))


def warp(field, slope):
    if slope == 0:
        return field
    side = len(field)
    q = np.arange(side)
    mapping = ((q[:, None] - slope * q[None, :]) % side)[..., None]
    return np.take_along_axis(field, mapping, axis=0)


def infer_shear(field, x_locations):
    side = len(field)
    y, z = np.unravel_index(x_locations, (side, side))
    reads = field[:, y, z]
    changes = reads != np.roll(reads, 1, axis=0)
    locations = np.arange(side)[:, None]
    profiles = []
    for slope in CANDIDATES:
        aligned = changes[(locations + slope * y[None]) % side, np.arange(len(y))]
        curve = aligned.mean(axis=1)
        profiles.append(curve[:side//2] + curve[side//2:])
    return CANDIDATES[int(np.max(profiles, axis=1).argmax())]


class TiltedWorld:
    def __init__(self, world, side, slope):
        self.slopes = (0, slope)
        self.routing = routing(world, side)
        self.fitted = []

    def distort(self, faults):
        return np.stack([warp(pulse, slope) for pulse, slope in zip(faults, self.slopes, strict=True)])

    def observe(self, faults, previous, known, probes, distance):
        policy_votes = [[], [], []]
        estimated_shifts = []
        estimated_slopes = []
        for current, active, direct_shift, true_slope in zip(faults, previous, known, self.slopes, strict=True):
            fitted = infer_shear(active, probes[0])
            estimated_slopes.append(fitted)
            for i, slope in enumerate((0, fitted, true_slope)):
                restored = warp(active, -slope)
                offset = direct_shift if i == 2 else infer_with_probes(restored, probes)
                axes = self.routing(restored, offset, probes, distance)
                local = witness_field(warp(current, -slope), None, offset, distance, selected=axes)
                policy_votes[i].append(warp(local, slope))
                if i == 1:
                    estimated_shifts.append(offset)
        self.fitted.append(estimated_slopes)
        return np.array(policy_votes), np.stack(estimated_shifts)


def paired(new, other):
    delta = new - other
    indices = np.random.default_rng(562218).integers(len(delta), size=(5000, len(delta)))
    bounds = np.percentile(delta[indices].mean(axis=1), [2.5, 97.5], axis=0)
    return dict(mean=delta.mean(axis=0).tolist(), interval=bounds.T.tolist())


def benchmark(driver, side, worlds, slopes, rounds=16, distance=8):
    records, layouts, inferred = [], [], []
    for world, slope in zip(worlds, slopes, strict=True):
        geometry = TiltedWorld(world, side, int(slope))
        counts, shifts = evaluate(driver, world, side=side, rounds=rounds, distance=distance,
                                  stratified=True, geometry=geometry)
        records.append(counts)
        layouts.append(shifts)
        inferred.append(geometry.fitted[1::2])
        print(f"Completed tilted world {world}", flush=True)
    return tuple(np.asarray(obj) for obj in (records, layouts, inferred))


def main():
    start = time.perf_counter()
    mx.set_cache_limit(2 * 1024**3)
    checkpoint = SEEDS[0]
    weights = (HERE / "results" / f"cube-full-{checkpoint}.json").read_bytes()
    driver = recorded_step(json.loads(weights))
    if sys.argv[1:] == ["--smoke"]:
        coords = np.arange(16)
        shape = np.broadcast_to(((coords[:, None, None] - 3) % 16 < 8), (16, 16, 16))
        samples = distributed_positions(np.random.default_rng(932144), 16)
        for slope in CANDIDATES:
            tilted = warp(shape, slope)
            assert infer_shear(tilted, samples) == slope
            np.testing.assert_array_equal(warp(tilted, -slope), shape)
        counts, shifts, fit = benchmark(driver, 16, WORLDS[:2], [-1, 1], rounds=4, distance=2)
        assert counts.shape == (2, 4, 3, 2, 6) and shifts.shape == (2, 4, 2, 2, 3)
        assert fit.shape == (2, 4, 2)
        assert (counts[..., 2] == 16**3).all()
        print("Verified invertible wall mapping, exact sign on reference planes and full 16³ MLX rollout")
        return
    out = Path(os.environ["STUDIO_JOB_DIR"])
    slopes = np.random.default_rng(223247).permutation([-1]*4 + [1]*4)
    counts, shifts, fit = benchmark(driver, SIDE, WORLDS, slopes)
    np.testing.assert_array_equal(counts[..., 4], np.broadcast_to(counts[:, :, :1, :, 4], counts[..., 4].shape))
    half = counts.shape[1] // 2
    late = counts[:, half:].sum(axis=1)
    rates = late[..., 3] / late[..., 2]
    to_axis = paired(rates[:, 1], rates[:, 0])
    to_true = paired(rates[:, 1], rates[:, 2])
    shifted_accuracy = float((fit[:, half:, 1] == slopes[:, None]).mean())
    legitimate = np.min(late[..., 4].sum(axis=0) / late[..., 0].sum(axis=0))
    passes = (shifted_accuracy >= .8 and to_axis["mean"][1] >= .02 and to_axis["interval"][1][0] > 0
              and to_true["interval"][1][0] >= -.015 and to_axis["interval"][0][0] >= -.005 and legitimate >= .98)
    result = summarize(counts[None], shifts[None], SIDE)
    result.update(decision="sparse_shear_adaptation_recovers_words" if passes else "unresolved",
                  worlds=list(WORLDS), true_shears=slopes.tolist(), cells=SIDE**3,
                  shifted_shear_accuracy=shifted_accuracy, adaptive_minus_axis_aligned=to_axis,
                  adaptive_minus_supplied=to_true, max_sample_fraction=3*PROBES/SIDE**2+4*8*ANCHORS/SIDE**3,
                  checkpoint=checkpoint, checkpoint_sha256=sha256(weights).hexdigest(),
                  seconds=time.perf_counter()-start, peak_mlx_bytes=mx.get_peak_memory())
    np.savez_compressed(out / "sheared.npz", counts=counts, shifts=shifts, fitted_shears=fit)
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
