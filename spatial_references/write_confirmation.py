"""Replay two timings of matched false write pulses in frozen spatial controllers."""
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import mlx.core as mx
import numpy as np

from long_hold import HERE, SEEDS
from volume import neighbors

MODES = ("single_hard", "double_soft", "double_hard")
STIMULI = ("separated", "bundled")
ROUNDS = 16
TICKS = 16
SIDE = 32
WORLDS = 24


def episodes(rng, side, rounds):
    shape = (1, side, side, side)
    for _ in range(rounds):
        valid = rng.random(shape) < .125
        faults = rng.random((1, side // 4, side // 4, side // 4)) < .25
        for axis in (1, 2, 3):
            faults = faults.repeat(4, axis=axis)
        actions = rng.integers(7, size=(TICKS, *shape), dtype=np.int32)
        blue = rng.integers(2, size=(TICKS, *shape), dtype=np.int8)
        yield valid, faults, actions, blue


def make_step(weights):
    read = mx.softmax(mx.array(weights["read"], dtype=mx.float32), axis=1)
    affine = mx.array(weights["gate"], dtype=mx.float32)
    bias = mx.array(weights["bias"], dtype=mx.float32)

    @mx.compile
    def step(work, memory, previous, command, blue, local):
        p = read[command, :7]
        new_work = sum(p[..., i:i + 1] * value for i, value in enumerate(neighbors(work, mx)))
        features = mx.stack(neighbors(blue, mx), axis=-1)
        score = local * affine[0] + features @ affine[1:] + bias
        soft = mx.sigmoid(score)
        positive = score > 0
        single, convex, double = memory
        rate = soft * previous
        return new_work, mx.stack((
            mx.where(positive[..., None], work, single),
            convex + rate[..., None] * (work - convex),
            mx.where((positive & (previous > .5))[..., None], work, double),
        )), soft

    @mx.compile
    def reads(work, memory, wanted, active):
        movement = sum(read[7, i] * value for i, value in enumerate(neighbors(work, mx)))
        matches = ((movement[None] + read[7, 7] * memory) > 0) == wanted
        return (matches.all(axis=-1) & active).sum(axis=(2, 3, 4))

    return step, reads


def evaluate(weights, world, side=SIDE, rounds=ROUNDS):
    rng = np.random.default_rng(721836 + world)
    initial = rng.integers(2, size=(side**3, 32), dtype=np.int8)
    shape = (1, side, side, side)
    words = initial.reshape(*shape, 32)
    work = mx.broadcast_to(mx.array(words, dtype=mx.float32) * 2 - 1, (2, *shape[1:], 32))
    memory = mx.stack([work] * len(MODES))
    previous = mx.zeros((2, *shape[1:]))
    address = reference = np.arange(side**3, dtype=np.int32).reshape(shape)
    step, reads = weights
    rounds_data = []

    for valid, faults, actions, blue in episodes(rng, side, rounds):
        data = np.zeros((2, len(MODES), 4), dtype=np.int32)
        for tick, (action, base) in enumerate(zip(actions, blue, strict=True), start=1):
            clock = ((3, 4), (9, 11), (9, 10))
            pulse = np.concatenate([
                valid if tick in clock[0] else faults if tick in window else np.zeros_like(valid)
                for window in clock[1:]
            ])
            b = np.where(np.roll(pulse, -1, axis=3), 1, base)
            local = np.where(pulse, 1, 1 - np.roll(b, 1, axis=3))
            assert np.array_equal(local * np.roll(b, 1, axis=3), pulse)
            work, memory, previous = step(work, memory, previous, mx.array(action),
                                          mx.array(b, dtype=mx.float32), mx.array(local, dtype=mx.float32))
            mx.eval(work, memory, previous)
            if tick == 4:
                reference = np.where(valid, address, reference)
            address = np.choose(action, neighbors(address, np))
            if tick in (4, TICKS):
                selected = valid if tick == 4 else np.ones_like(valid)
                wanted = mx.array(initial[reference], dtype=mx.bool_)
                column = 0 if tick == 4 else 2
                data[..., column] = selected.sum()
                data[..., column + 1] = np.array(reads(work, memory, wanted, mx.array(selected))).T
        rounds_data.append(data)
    return np.stack(rounds_data)


def summarize(data):
    paired = data[:, :, data.shape[2] // 2:].sum(axis=(0, 2))
    totals = paired.sum(axis=0)
    valid = totals[..., 1] / totals[..., 0]
    after = totals[..., 3] / totals[..., 2]
    draws = np.random.default_rng(832859).integers(len(paired), size=(4000, len(paired)))
    draws_totals = paired[draws].sum(axis=1)
    draw_rates = draws_totals[..., 3] / draws_totals[..., 2]
    gain = after[0, 2] - after[0, 0]
    ci = np.percentile(draw_rates[:, 0, 2] - draw_rates[:, 0, 0], [2.5, 97.5])
    robust = min(valid[0, 2], after[0, 2]) >= .95 and gain >= .05 and ci[0] > 0
    return {
        "decision": "separated_pulses_rejected" if robust else "fault_tolerance_unresolved",
        "per_stimulus": {
            stimulus: {
                mode: {"after_valid_write": float(valid[i, j]), "after_fault_interval": float(after[i, j])}
                for j, mode in enumerate(MODES)
            } for i, stimulus in enumerate(STIMULI)
        },
        "double_minus_single_separated": {"mean": float(gain), "interval": ci.tolist()},
        "soft_competitive_separated": bool(max(valid[0, 2] - valid[0, 1], after[0, 2] - after[0, 1]) <= .02),
        "counts": totals.tolist(),
    }


def main():
    start = time.perf_counter()
    output = Path(os.environ["STUDIO_JOB_DIR"])
    mx.set_cache_limit(2 * 1024**3)
    runs = []
    digests = {}
    for seed in SEEDS:
        checkpoint = HERE / "results" / f"cube-full-{seed}.json"
        digests[str(seed)] = sha256(checkpoint.read_bytes()).hexdigest()
        step = make_step(json.loads(checkpoint.read_text()))
        runs.append(np.stack([evaluate(step, world) for world in range(WORLDS)]))
        print(seed, summarize(np.stack(runs[-1:]))["per_stimulus"], flush=True)
    data = np.stack(runs)
    np.save(output / "per-world.npy", data)
    result = dict(shape=list(data.shape), voxels=SIDE**3, word_bits=32, transitions=ROUNDS*TICKS,
                  runtime_s=time.perf_counter() - start, peak_mlx_bytes=mx.get_peak_memory(),
                  frozen_checkpoints=digests, **summarize(data))
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
