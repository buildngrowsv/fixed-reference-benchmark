"""Route a single confirmation sensor using recent primary detector co-activation."""
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import mlx.core as mx
import numpy as np

from long_hold import HERE, SEEDS
from spatial_witness import fault_mask, recorded_step
from volume import neighbors

SIDE = 64
ROUNDS = 16
WORLDS = 12
POLICIES = ("fixed", "calibrated", "adaptive", "oracle")
ENVIRONMENTS = ("stable", "rotated")


def overlaps(active, distance):
    return np.column_stack([
        (active & np.roll(active, distance, axis=axis)).mean(axis=(1, 2, 3))
        for axis in (1, 2, 3)
    ])


def evaluate(compiled, world, side=SIDE, rounds=ROUNDS, distance=8):
    rng = np.random.default_rng(862017 + world)
    initial_axis = (world // 3) % 3
    next_axis = (initial_axis + rng.integers(1, 3)) % 3
    baseline = world % 3
    shape = (1, side, side, side)
    size = (len(ENVIRONMENTS), *shape[1:])
    source = rng.integers(2, size=(side**3, 32), dtype=np.int8).astype(bool)
    work = mx.array(source.reshape(*shape, 32), dtype=mx.float32) * 2 - 1
    memory = mx.broadcast_to(work, (len(POLICIES), *size, 32))
    previous = mx.zeros(size, dtype=mx.bool_)
    address = saved = np.arange(side**3, dtype=np.int32).reshape(shape)
    step, measure = compiled
    recent = np.zeros((2, 3), dtype=np.float64)
    training = recent.copy()
    static = np.full(2, baseline)
    unrestricted = np.ones((len(POLICIES), *size), dtype=bool)
    frames = []
    selected_axes = []
    volumes = np.full(2, side**3 // 4)

    for round_index in range(rounds):
        selected = rng.random(shape) < .125
        delivered = selected & (rng.random(shape) >= .01)
        true_witness = np.broadcast_to(delivered, unrestricted.shape)
        base = fault_mask(rng, side)[1]
        hidden_axis = np.array([initial_axis, next_axis if round_index >= rounds // 2 else initial_axis])
        faults = np.stack([np.moveaxis(base, 2, axis) for axis in hidden_axis])
        np.testing.assert_array_equal(faults.sum(axis=(1, 2, 3)), volumes)
        witnesses = np.stack([np.roll(faults, distance, axis=axis) for axis in (1, 2, 3)])
        data = np.zeros((len(POLICIES), len(ENVIRONMENTS), 6), dtype=np.int32)

        for tick in range(1, 17):
            command = rng.integers(7, size=shape, dtype=np.int32)
            blue = rng.integers(2, size=shape, dtype=np.int8)
            witness = unrestricted
            if tick in (3, 4):
                signal = np.broadcast_to(selected, size)
                witness = true_witness
            elif tick in (9, 10):
                signal = faults
                picks = np.stack((np.full(2, baseline), static, recent.argmin(axis=1), (hidden_axis + 1) % 3))
                witness = witnesses[picks, np.arange(2)]
            else:
                signal = np.zeros(size, dtype=bool)
            blue = np.where(np.roll(signal, -1, axis=-1), 1, blue)
            local = np.where(signal, 1, 1 - np.roll(blue, 1, axis=-1))
            work, memory, previous, writes = step(work, memory, previous, mx.array(command),
                                                 mx.array(blue, dtype=mx.float32), mx.array(local, dtype=mx.float32),
                                                 mx.array(witness))
            mx.eval(work, memory, previous, writes)
            observation = overlaps(np.array(previous), distance)
            recent = .92 * recent + .08 * observation
            if round_index < 4:
                training += observation
            if tick == 4:
                saved = np.where(selected, address, saved)
                data[..., 4] = np.array(writes)
            if tick == 10:
                selected_axes.append(np.vstack((picks, hidden_axis)))
                data[..., 5] = np.array(writes)
            address = np.choose(command, neighbors(address, np))
            if tick in (4, 16):
                eligible = selected if tick == 4 else np.ones_like(selected)
                at = 0 if tick == 4 else 2
                data[..., at] = eligible.sum()
                data[..., at + 1] = np.array(measure(work, memory, mx.array(source[saved]), mx.array(eligible)))
        if round_index == 3:
            static = training.argmin(axis=1)
        frames.append(data)

    return np.stack(frames), np.stack(selected_axes)


def summarize(data, choices):
    last = data.shape[2] // 2
    by_world = data[:, :, last:].sum(axis=(0, 2))
    totals = by_world.sum(axis=0)
    rates = totals[..., 3] / totals[..., 2]
    accepts = totals[..., 4] / totals[..., 0]
    draws = np.random.default_rng(571039).integers(len(by_world), size=(4000, len(by_world)))
    sample = by_world[draws].sum(axis=1)
    scores = sample[..., 3] / sample[..., 2]
    deltas = {name: rates[2] - rates[i] for i, name in enumerate(POLICIES) if i != 2}
    intervals = {name: np.percentile(scores[:, 2] - scores[:, i], [2.5, 97.5], axis=0)
                 for i, name in enumerate(POLICIES) if i != 2}
    test = all(deltas[p][1] >= .05 and intervals[p][0, 1] > 0 for p in ("fixed", "calibrated"))
    test = test and deltas["calibrated"][0] >= -.02 and accepts[2].min() >= .98
    chosen = choices[:, :, last:]
    aligns = (chosen[..., :len(POLICIES), :] == chosen[..., -1:, :]).mean(axis=(0, 1, 2))
    result = {name: {env: dict(retention=float(rates[i, j]), authorized=float(accepts[i, j]),
                                      false_writes=float(totals[i, j, 5] / totals[i, j, 2]),
                                      chooses_fault_axis=float(aligns[i, j]))
                            for j, env in enumerate(ENVIRONMENTS)} for i, name in enumerate(POLICIES)}
    contrast = {p: {env: dict(mean=float(delta[j]), interval=intervals[p][:, j].tolist())
                    for j, env in enumerate(ENVIRONMENTS)} for p, delta in deltas.items()}
    return dict(decision="adapts_to_rotated_faults" if test else "unresolved",
                results=result, contrasts=contrast, counts=totals.tolist())


def main():
    start = time.perf_counter()
    out = Path(os.environ["STUDIO_JOB_DIR"])
    mx.set_cache_limit(2 * 1024**3)
    data, choices, digest = [], [], {}
    for seed in SEEDS:
        path = HERE / "results" / f"cube-full-{seed}.json"
        digest[str(seed)] = sha256(path.read_bytes()).hexdigest()
        step = recorded_step(json.loads(path.read_text()))
        rows, axes = map(np.stack, zip(*(evaluate(step, w) for w in range(WORLDS)), strict=True))
        data.append(rows)
        choices.append(axes)
        print(f"Completed {seed}: {len(rows)} worlds", flush=True)
    samples, axes = np.stack(data), np.stack(choices)
    np.savez_compressed(out / "worlds.npz", counts=samples, axes=axes)
    result = dict(voxels=SIDE**3, worlds=WORLDS, transitions=16*ROUNDS, frozen_checkpoints=digest,
                  runtime_s=time.perf_counter() - start, peak_mlx_bytes=mx.get_peak_memory(), **summarize(samples, axes))
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
