"""Compare one global witness direction against independent regional directions."""
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import mlx.core as mx
import numpy as np

from adaptive_witness import overlaps
from long_hold import HERE, SEEDS
from spatial_witness import fault_mask, recorded_step
from volume import neighbors

SIDE = 96
WORLDS = 8
ROUNDS = 16
POLICIES = ("global", "regional", "orientation")
FIELDS = ("homogeneous", "heterogeneous")


def divide(values):
    width = values.shape[-1] // 2
    prefix = values.shape[:-3]
    d = len(prefix)
    split = values.reshape(*prefix, 2, width, 2, width, 2, width)
    order = (*range(d), d, d+2, d+4, d+1, d+3, d+5)
    return split.transpose(order).reshape(*prefix, 8, width, width, width)


def combine(values):
    width = values.shape[-1]
    prefix = values.shape[:-4]
    d = len(prefix)
    block = values.reshape(*prefix, 2, 2, 2, width, width, width)
    order = (*range(d), d, d+3, d+1, d+4, d+2, d+5)
    return block.transpose(order).reshape(*prefix, 2*width, 2*width, 2*width)


def evaluate(compiled, world, side=SIDE, rounds=ROUNDS, distance=8):
    rng = np.random.default_rng(673101 + world)
    shape = (1, side, side, side)
    size = (2, *shape[1:])
    half = side // 2
    source = rng.integers(2, size=(side**3, 32), dtype=np.int8).astype(bool)
    work = mx.array(source.reshape(*shape, 32), dtype=mx.float32) * 2 - 1
    memory = mx.broadcast_to(work, (len(POLICIES), *size, 32))
    previous = mx.zeros(size, dtype=mx.bool_)
    address = saved = np.arange(side**3, dtype=np.int32).reshape(shape)
    step, measure = compiled
    recent = np.zeros((2, 8, 3))
    unrestricted = np.ones((len(POLICIES), *size), dtype=bool)
    axis = np.stack((np.full(8, world % 3), rng.permutation(np.repeat([0, 1, 2], [3, 3, 2]))))
    results, choices = [], []
    field_indices = np.arange(2)[None, :, None]
    region_indices = np.arange(8)

    for _ in range(rounds):
        selected = rng.random(shape) < .125
        delivered = selected & (rng.random(shape) >= .01)
        true_witness = np.broadcast_to(delivered, unrestricted.shape)
        base = [fault_mask(rng, half)[1] for _ in range(8)]
        region_faults = np.stack([
            np.stack([np.moveaxis(template, 2, direction) for template, direction in zip(base, layout_axes, strict=True)])
            for layout_axes in axis
        ])
        assert (region_faults.sum(axis=(-3, -2, -1)) == half**3 // 4).all()
        faults = combine(region_faults)
        witnesses = np.stack([np.roll(region_faults, distance, axis=i) for i in (-3, -2, -1)])
        data = np.zeros((len(POLICIES), 2, 6), dtype=np.int32)

        for tick in range(1, 17):
            command = rng.integers(7, size=shape, dtype=np.int32)
            blue = rng.integers(2, size=shape, dtype=np.int8)
            witness = unrestricted
            if tick in (3, 4):
                signal = np.broadcast_to(selected, size)
                witness = true_witness
            elif tick in (9, 10):
                signal = faults
                shared = recent.mean(1).argmin(1)
                local = recent.argmin(2)
                routes = np.stack((np.broadcast_to(shared[:, None], local.shape), local, (axis + 1) % 3))
                witness = combine(witnesses[routes, field_indices, region_indices])
            else:
                signal = np.zeros(size, dtype=bool)
            blue = np.where(np.roll(signal, -1, axis=-1), 1, blue)
            latch = np.where(signal, 1, 1 - np.roll(blue, 1, axis=-1))
            work, memory, previous, writes = step(work, memory, previous, mx.array(command),
                                                 mx.array(blue, dtype=mx.float32), mx.array(latch, dtype=mx.float32),
                                                 mx.array(witness))
            mx.eval(work, memory, previous, writes)
            observations = divide(np.array(previous)).reshape(-1, half, half, half)
            recent = .92 * recent + .08 * overlaps(observations, distance).reshape(2, 8, 3)
            if tick == 4:
                saved = np.where(selected, address, saved)
                data[..., 4] = np.array(writes)
            if tick == 10:
                choices.append(np.concatenate((routes, axis[None]), axis=0))
                data[..., 5] = np.array(writes)
            address = np.choose(command, neighbors(address, np))
            if tick in (4, 16):
                eligible = selected if tick == 4 else np.ones_like(selected)
                at = 0 if tick == 4 else 2
                data[..., at] = eligible.sum()
                data[..., at + 1] = np.array(measure(work, memory, mx.array(source[saved]), mx.array(eligible)))
        results.append(data)
    return np.stack(results), np.stack(choices)


def summarize(data, choices):
    last = data.shape[2] // 2
    world = data[:, :, last:].sum(axis=(0, 2))
    totals = world.sum(axis=0)
    reads = totals[..., 3] / totals[..., 2]
    writes = totals[..., 4] / totals[..., 0]
    aligned = (choices[:, :, last:, :3] == choices[:, :, last:, 3:]).mean(axis=(0, 1, 2, 5))
    draws = np.random.default_rng(789865).integers(len(world), size=(4000, len(world)))
    sampled = world[draws].sum(axis=1)
    rates = sampled[..., 3] / sampled[..., 2]
    advance = reads[1] - reads[0]
    ci = np.percentile(rates[:, 1] - rates[:, 0], [2.5, 97.5], axis=0)
    passes = advance[1] >= .05 and ci[0, 1] > 0 and advance[0] >= -.02 and writes[1].min() >= .98
    ret = {name: {field: dict(retention=float(reads[i, j]), legitimate=float(writes[i, j]),
                             false_writes=float(totals[i, j, 5] / totals[i, j, 2]),
                             chooses_fault_axis=float(aligned[i, j]))
                  for j, field in enumerate(FIELDS)} for i, name in enumerate(POLICIES)}
    return dict(decision="regional_beats_global" if passes else "unresolved", rates=ret, counts=totals.tolist(),
                regional_minus_global={field: dict(mean=float(advance[j]), interval=ci[:, j].tolist())
                                       for j, field in enumerate(FIELDS)})


def main():
    start = time.perf_counter()
    out = Path(os.environ["STUDIO_JOB_DIR"])
    mx.set_cache_limit(2 * 1024**3)
    counts, choices, digests = [], [], {}
    for seed in SEEDS:
        path = HERE / "results" / f"cube-full-{seed}.json"
        digests[str(seed)] = sha256(path.read_bytes()).hexdigest()
        step = recorded_step(json.loads(path.read_text()))
        tallies, axes = map(np.stack, zip(*(evaluate(step, w) for w in range(WORLDS)), strict=True))
        counts.append(tallies)
        choices.append(axes)
        print(f"Completed {seed}: {len(tallies)} worlds", flush=True)
    counts, choices = np.stack(counts), np.stack(choices)
    np.savez_compressed(out / "worlds.npz", counts=counts, choices=choices)
    result = dict(voxels=SIDE**3, worlds=WORLDS, transitions=ROUNDS*16, frozen_checkpoints=digests,
                  runtime_s=time.perf_counter() - start, peak_mlx_bytes=mx.get_peak_memory(),
                  **summarize(counts, choices))
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
