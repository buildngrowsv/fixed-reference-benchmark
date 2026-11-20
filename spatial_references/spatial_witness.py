"""Test how the physical distance between write confirmations affects retained state."""
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import mlx.core as mx
import numpy as np

from long_hold import HERE, SEEDS
from volume import neighbors

SIDE = 64
WORLDS = 8
ROUNDS = 16
STIMULI = ("tiles", "columns")
POLICIES = ("two_ticks", "near", "far")


def fault_mask(rng, side):
    tiles = side // 4
    def sampled(n):
        values = np.zeros(n, dtype=bool)
        values[rng.choice(n, n // 4, replace=False)] = True
        return values

    cubes = sampled(tiles**3).reshape(tiles, tiles, tiles)
    for axis in range(3):
        cubes = np.repeat(cubes, 4, axis=axis)
    layer = sampled(tiles**2).reshape(tiles, tiles).repeat(4, axis=0).repeat(4, axis=1)
    columns = np.broadcast_to(layer[..., None], cubes.shape)
    return np.stack((cubes, columns))


def recorded_step(weights):
    read = mx.softmax(mx.array(weights["read"], dtype=mx.float32), axis=1)
    coefficients = mx.array(weights["gate"], dtype=mx.float32)
    bias = mx.array(weights["bias"], dtype=mx.float32)

    @mx.compile
    def step(work, memory, prior, command, blue, local, witness):
        p = read[command, :7]
        advance = sum(p[..., i:i + 1] * value for i, value in enumerate(neighbors(work, mx)))
        surroundings = mx.stack(neighbors(blue, mx), axis=-1)
        active = local * coefficients[0] + surroundings @ coefficients[1:] + bias > 0
        confirmed = witness & active & prior
        updated = mx.where(confirmed[..., None], work, memory)
        return advance, updated, active, confirmed.sum(axis=(2, 3, 4))

    @mx.compile
    def measure(work, memory, expected, eligible):
        remainder = sum(read[7, i] * v for i, v in enumerate(neighbors(work, mx)))
        matches = ((remainder + read[7, 7] * memory) > 0) == expected
        return (matches.all(axis=-1) & eligible).sum(axis=(2, 3, 4))

    return step, measure


def evaluate(compiled, seed, side=SIDE, rounds=ROUNDS, far=8):
    rng = np.random.default_rng(582412 + seed)
    shape = (1, side, side, side)
    source = rng.integers(2, size=(side**3, 32), dtype=np.int8).astype(bool)
    work = mx.array(source.reshape(*shape, 32), dtype=mx.float32) * 2 - 1
    memory = mx.broadcast_to(work, (len(POLICIES), len(STIMULI), *shape[1:], 32))
    prior = mx.zeros((len(STIMULI), *shape[1:]), dtype=mx.bool_)
    address = intended = np.arange(side**3, dtype=np.int32).reshape(shape)
    step, measure = compiled
    records = []

    for _ in range(rounds):
        selected = rng.random(shape) < .125
        received = selected & (rng.random(shape) >= .01)
        faults = fault_mask(rng, side)
        witnesses = [np.roll(faults, distance, axis=-1) for distance in (1, far)]
        np.testing.assert_array_equal(faults.sum(axis=(1, 2, 3)), np.full(2, side**3 // 4))
        assert np.array_equal(witnesses[0][1], witnesses[1][1])
        unrestricted = np.ones_like(faults)
        authorized = np.broadcast_to(received, faults.shape)
        data = np.zeros((len(POLICIES), len(STIMULI), 6), dtype=np.int32)
        for tick in range(1, 17):
            action = rng.integers(7, size=shape, dtype=np.int32)
            blue = rng.integers(2, size=shape, dtype=np.int8)
            if tick in (3, 4):
                signal = np.broadcast_to(selected, faults.shape)
                observed = np.stack((unrestricted, authorized, authorized))
            elif tick in (9, 10):
                signal = faults
                observed = np.stack((unrestricted, *witnesses))
            else:
                signal = np.zeros_like(faults)
                observed = np.ones((len(POLICIES), *faults.shape), dtype=bool)
            blue = np.where(np.roll(signal, -1, axis=-1), 1, blue)
            local = np.where(signal, 1, 1 - np.roll(blue, 1, axis=-1))
            work, memory, prior, writes = step(work, memory, prior, mx.array(action),
                                               mx.array(blue, dtype=mx.float32),
                                               mx.array(local, dtype=mx.float32), mx.array(observed))
            mx.eval(work, memory, prior, writes)
            if tick == 4:
                intended = np.where(selected, address, intended)
                data[..., 4] = np.array(writes)
            if tick == 10:
                data[..., 5] = np.array(writes)
            address = np.choose(action, neighbors(address, np))
            if tick in (4, 16):
                eligible = selected if tick == 4 else np.ones_like(selected)
                at = 0 if tick == 4 else 2
                data[..., at] = eligible.sum()
                data[..., at + 1] = np.array(measure(work, memory, mx.array(source[intended]), mx.array(eligible)))
        records.append(data)
    return np.stack(records)


def summarize(data):
    by_world = data[:, :, data.shape[2] // 2:].sum(axis=(0, 2))
    totals = by_world.sum(axis=0)
    read = totals[..., 3] / totals[..., 2]
    write = totals[..., 4] / totals[..., 0]
    draws = np.random.default_rng(727143).integers(len(by_world), size=(4000, len(by_world)))
    sampled = by_world[draws].sum(axis=1)
    read_draw = sampled[..., 3] / sampled[..., 2]
    advantage = read[2] - read[1]
    ci = np.percentile(read_draw[:, 2] - read_draw[:, 1], [2.5, 97.5], axis=0)
    passes = advantage[0] >= .05 and ci[0, 0] > 0 and abs(advantage[1]) <= .02 and write[1:].min() >= .98
    return dict(decision="geometry_matters" if passes else "unresolved", counts=totals.tolist(),
                rates={p: {s: dict(authorized_acceptance=float(write[j, i]),
                                   after_legitimate=float(totals[j, i, 1] / totals[j, i, 0]),
                                   after_faults=float(read[j, i]),
                                   false_write_fraction=float(totals[j, i, 5] / totals[j, i, 2]))
                           for i, s in enumerate(STIMULI)} for j, p in enumerate(POLICIES)},
                far_minus_near={s: dict(mean=float(advantage[i]), interval=ci[:, i].tolist())
                                for i, s in enumerate(STIMULI)})


def main():
    start = time.perf_counter()
    output = Path(os.environ["STUDIO_JOB_DIR"])
    mx.set_cache_limit(2 * 1024**3)
    data = []
    digests = {}
    for seed in SEEDS:
        checkpoint = HERE / "results" / f"cube-full-{seed}.json"
        digests[str(seed)] = sha256(checkpoint.read_bytes()).hexdigest()
        compiled = recorded_step(json.loads(checkpoint.read_text()))
        data.append(np.stack([evaluate(compiled, world) for world in range(WORLDS)]))
        print(seed, summarize(np.stack(data[-1:]))["far_minus_near"], flush=True)
    result = np.stack(data)
    np.save(output / "per-world.npy", result)
    report = dict(physical_shape=[SIDE] * 3, voxels=SIDE**3, transitions=ROUNDS * 16,
                  worlds=WORLDS, word_bits=32, frozen_checkpoints=digests,
                  runtime_s=time.perf_counter() - start, peak_mlx_bytes=mx.get_peak_memory(),
                  **summarize(result))
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
