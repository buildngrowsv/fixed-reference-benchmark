"""Infer translated region walls from previous-tick sensor disagreements."""
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import mlx.core as mx
import numpy as np

from adaptive_witness import overlaps
from long_hold import HERE, SEEDS
from regional_witness import combine, divide
from spatial_witness import fault_mask, recorded_step
from volume import neighbors

SIDE = 96
WORLDS = 8
ROUNDS = 16
POLICIES = ("fixed", "inferred", "known")
STIMULI = ("aligned", "shifted")
AXES = (-3, -2, -1)


def infer_boundaries(active):
    half = active.shape[0] // 2
    cuts = []
    for axis in range(3):
        neighbors_other = tuple(i for i in range(3) if i != axis)
        transitions = (active != np.roll(active, 1, axis=axis)).mean(axis=neighbors_other)
        cuts.append((transitions[:half] + transitions[half:]).argmax())
    return np.array(cuts)


def witness_field(current, previous, shift, distance):
    sensed = divide(np.roll(previous, -shift, axis=AXES))
    selected = overlaps(sensed, distance).argmin(axis=1)
    inside = divide(np.roll(current, -shift, axis=AXES))
    choices = np.stack([np.roll(inside, distance, axis=i) for i in AXES])
    votes = choices[selected, np.arange(8)]
    return np.roll(combine(votes), shift, axis=AXES)


def evaluate(compiled, world, side=SIDE, rounds=ROUNDS, distance=8):
    rng = np.random.default_rng(931622 + world)
    shape = (1, side, side, side)
    size = (2, *shape[1:])
    half = side // 2
    assert half >= 8 and half % 4 == 0
    bits = rng.integers(2, size=(side**3, 32), dtype=np.int8).astype(bool)
    work = mx.array(bits.reshape(*shape, 32), dtype=mx.float32) * 2 - 1
    previous = mx.zeros(size, dtype=mx.bool_)
    memory = mx.broadcast_to(work, (len(POLICIES), *size, 32))
    origin = saved = np.arange(side**3, dtype=np.int32).reshape(shape)
    known = np.stack((np.zeros(3, dtype=int), rng.choice(np.arange(4, half, 4), size=3)))
    directions = rng.permutation(np.repeat([0, 1, 2], [3, 3, 2]))
    step, measure = compiled
    unrestricted = np.ones((len(POLICIES), *size), dtype=bool)
    counts, shifts = [], []

    for _ in range(rounds):
        selected = rng.random(shape) < .125
        delivered = selected & (rng.random(shape) >= .01)
        regions = [np.moveaxis(fault_mask(rng, half)[1], 2, a) for a in directions]
        base = combine(np.stack(regions))
        faults = np.stack([np.roll(base, shift, axis=AXES) for shift in known])
        assert (faults.sum(axis=AXES) == side**3 // 4).all()
        data = np.zeros((len(POLICIES), len(STIMULI), 6), dtype=np.int32)

        for tick in range(1, 17):
            movement = rng.integers(7, size=shape, dtype=np.int32)
            base_blue = rng.integers(2, size=shape, dtype=np.int8)
            if tick in (3, 4):
                pulses = np.broadcast_to(selected, size)
                witness = np.broadcast_to(delivered, unrestricted.shape)
            elif tick in (9, 10):
                pulses = faults
                observed = np.array(previous)
                inferred = np.stack([infer_boundaries(active) for active in observed])
                offsets = np.stack((np.zeros_like(known), inferred, known))
                witness = np.stack([np.stack([
                    witness_field(fault, act, shift, distance)
                    for fault, act, shift in zip(faults, observed, arm, strict=True)
                ]) for arm in offsets])
            else:
                pulses = np.zeros(size, dtype=bool)
                witness = unrestricted
            blue = np.where(np.roll(pulses, -1, axis=-1), 1, base_blue)
            local = np.where(pulses, 1, 1 - np.roll(blue, 1, axis=-1))
            work, memory, previous, writes = step(work, memory, previous, mx.array(movement),
                                                mx.array(blue, dtype=mx.float32),
                                                mx.array(local, dtype=mx.float32), mx.array(witness))
            mx.eval(work, memory, previous, writes)
            if tick == 4:
                saved = np.where(selected, origin, saved)
                data[..., 4] = np.array(writes)
            if tick == 10:
                shifts.append(np.stack((inferred, known)))
                data[..., 5] = np.array(writes)
            origin = np.choose(movement, neighbors(origin, np))
            if tick in (4, 16):
                valid = selected if tick == 4 else np.ones_like(selected)
                at = 0 if tick == 4 else 2
                data[..., at] = valid.sum()
                data[..., at+1] = np.array(measure(work, memory, mx.array(bits[saved]), mx.array(valid)))
        counts.append(data)
    return np.stack(counts), np.stack(shifts)


def summarize(counts, shifts, side):
    halfway = counts.shape[2] // 2
    worlds = counts[:, :, halfway:].sum(axis=(0, 2))
    totals = worlds.sum(axis=0)
    rates = totals[..., 3] / totals[..., 2]
    legit = totals[..., 4] / totals[..., 0]
    re = shifts[:, :, halfway:]
    shift_errors = np.abs(re[:, :, :, 0] - re[:, :, :, 1])
    shift_errors = np.minimum(shift_errors, side//2 - shift_errors)
    draws = np.random.default_rng(686132).integers(len(worlds), size=(4000, len(worlds)))
    sampled = worlds[draws].sum(axis=1)
    recovered = sampled[..., 3] / sampled[..., 2]
    delta = rates[1] - rates[0]
    limits = np.percentile(recovered[:, 1] - recovered[:, 0], [2.5, 97.5], axis=0)
    passes = delta[1] >= .05 and limits[0, 1] > 0 and delta[0] >= -.02 and legit[1].min() >= .98
    return dict(decision="inferred_boundaries_help" if passes else "unresolved", counts=totals.tolist(),
                rates={p: {s: dict(retention=float(rates[i, j]), legitimate=float(legit[i, j]),
                                   false_writes=float(totals[i, j, 5]/totals[i, j, 2]))
                            for j, s in enumerate(STIMULI)} for i, p in enumerate(POLICIES)},
                boundary_accuracy={s: dict(exact=float((shift_errors[..., j, :] == 0).mean()),
                                           within_four=float((shift_errors[..., j, :] <= 4).mean()))
                                   for j, s in enumerate(STIMULI)},
                inferred_minus_fixed={s: dict(mean=float(delta[j]), interval=limits[:, j].tolist())
                                       for j, s in enumerate(STIMULI)})


def main():
    start = time.perf_counter()
    out = Path(os.environ["STUDIO_JOB_DIR"])
    mx.set_cache_limit(2 * 1024**3)
    all_counts, all_shifts, hashes = [], [], {}
    for seed in SEEDS:
        path = HERE / "results" / f"cube-full-{seed}.json"
        hashes[str(seed)] = sha256(path.read_bytes()).hexdigest()
        compiled = recorded_step(json.loads(path.read_text()))
        counts, shifts = map(np.stack, zip(*(evaluate(compiled, w) for w in range(WORLDS)), strict=True))
        all_counts.append(counts)
        all_shifts.append(shifts)
        print(f"Completed checkpoint {seed}", flush=True)
    counts, shifts = np.stack(all_counts), np.stack(all_shifts)
    np.savez_compressed(out / "worlds.npz", counts=counts, shifts=shifts)
    summary = dict(voxels=SIDE**3, worlds=WORLDS, transitions=ROUNDS*16, frozen_checkpoints=hashes,
                   runtime_s=time.perf_counter()-start, peak_mlx_bytes=mx.get_peak_memory(),
                   **summarize(counts, shifts, SIDE))
    (out / "result.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
