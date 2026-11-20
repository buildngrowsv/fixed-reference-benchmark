"""Infer translated region walls from previous-tick sensor disagreements."""
import argparse
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
PROBES = 64
SOURCE = "spatial-inferred-regions-884736"


def infer_boundaries(active):
    half = active.shape[0] // 2
    cuts = []
    for axis in range(3):
        neighbors_other = tuple(i for i in range(3) if i != axis)
        transitions = (active != np.roll(active, 1, axis=axis)).mean(axis=neighbors_other)
        cuts.append((transitions[:half] + transitions[half:]).argmax())
    return np.array(cuts)


def infer_with_probes(active, locations):
    side = active.shape[0]
    half = side // 2
    cuts = []
    for axis, pixels in enumerate(locations):
        views = active.swapaxes(0, axis).reshape(side, -1)[:, pixels]
        disagreement = (views != np.roll(views, 1, axis=0)).mean(axis=1)
        cuts.append((disagreement[:half] + disagreement[half:]).argmax())
    return np.array(cuts)


def witness_field(current, previous, shift, distance, *, selected=None):
    if selected is None:
        sensed = divide(np.roll(previous, -shift, axis=AXES))
        selected = overlaps(sensed, distance).argmin(axis=1)
    inside = divide(np.roll(current, -shift, axis=AXES))
    choices = np.stack([np.roll(inside, distance, axis=i) for i in AXES])
    votes = choices[selected, np.arange(8)]
    return np.roll(combine(votes), shift, axis=AXES)


def distributed_positions(rng, side):
    strata = int(np.sqrt(PROBES))
    size = side // strata
    assert side % strata == 0 and strata**2 == PROBES
    a, b = np.mgrid[:strata, :strata]
    rows = size * a.ravel() + rng.integers(size, size=PROBES)
    columns = size * b.ravel() + rng.integers(size, size=PROBES)
    return rows*side + columns


def evaluate(compiled, world, side=SIDE, rounds=ROUNDS, distance=8, sparse=False, stratified=False,
             sampled_routes=None):
    rng = np.random.default_rng(931622 + world)
    observer = np.random.default_rng(790133 + world)
    probes = None
    if stratified:
        probes = [distributed_positions(observer, side) for _ in range(3)]
    elif sparse:
        probes = [observer.choice(side**2, size=PROBES, replace=False) for _ in range(3)]
    witness_rules = [witness_field] * len(POLICIES)
    if sampled_routes is not None:
        if not stratified:
            raise ValueError("sampled routing requires stratified probes")

        def sampled_witness(fault, active, shift, distance):
            selected = sampled_routes(active, shift, probes, distance)
            return witness_field(fault, active, shift, distance, selected=selected)

        witness_rules[POLICIES.index("inferred")] = sampled_witness
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
                inferred = np.stack([infer_boundaries(active) if probes is None
                                     else infer_with_probes(active, probes) for active in observed])
                offsets = np.stack((np.zeros_like(known), inferred, known))
                witness = np.stack([np.stack([
                    rule(fault, act, shift, distance)
                    for fault, act, shift in zip(faults, observed, arm, strict=True)
                ]) for rule, arm in zip(witness_rules, offsets, strict=True)])
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


def compare_sparse(archive, counts):
    with np.load(archive) as data:
        full = data["counts"]
    assert full.shape == counts.shape
    np.testing.assert_array_equal(full[:, :, :, (0, 2)], counts[:, :, :, (0, 2)])
    old = full[:, :, 8:, 1].sum(axis=(0, 2))
    new = counts[:, :, 8:, 1].sum(axis=(0, 2))
    candidates = np.random.default_rng(591340).integers(len(old), size=(4000, len(old)))
    a, b = new[candidates].sum(axis=1), old[candidates].sum(axis=1)
    runs = a[..., 3]/a[..., 2] - b[..., 3]/b[..., 2]
    sum_a, sum_b = new.sum(axis=0), old.sum(axis=0)
    change = sum_a[..., 3]/sum_a[..., 2] - sum_b[..., 3]/sum_b[..., 2]
    ci = np.percentile(runs, [2.5, 97.5], axis=0)
    return {s: dict(mean=float(change[i]), interval=ci[:, i].tolist()) for i, s in enumerate(STIMULI)}


def main():
    ap = argparse.ArgumentParser()
    observations = ap.add_mutually_exclusive_group()
    observations.add_argument("--sparse", action="store_true")
    observations.add_argument("--stratified", action="store_true")
    args = ap.parse_args()
    start = time.perf_counter()
    out = Path(os.environ["STUDIO_JOB_DIR"])
    mx.set_cache_limit(2 * 1024**3)
    all_counts, all_shifts, hashes = [], [], {}
    for seed in SEEDS:
        path = HERE / "results" / f"cube-full-{seed}.json"
        hashes[str(seed)] = sha256(path.read_bytes()).hexdigest()
        compiled = recorded_step(json.loads(path.read_text()))
        counts, shifts = map(np.stack, zip(*(evaluate(compiled, w, sparse=args.sparse, stratified=args.stratified) for w in range(WORLDS)), strict=True))
        all_counts.append(counts)
        all_shifts.append(shifts)
        print(f"Completed checkpoint {seed}", flush=True)
    counts, shifts = np.stack(all_counts), np.stack(all_shifts)
    np.savez_compressed(out / "worlds.npz", counts=counts, shifts=shifts)
    summary = dict(voxels=SIDE**3, worlds=WORLDS, transitions=ROUNDS*16, frozen_checkpoints=hashes,
                   runtime_s=time.perf_counter()-start, peak_mlx_bytes=mx.get_peak_memory(),
                   **summarize(counts, shifts, SIDE))
    if args.sparse or args.stratified:
        run_root = Path(os.environ["STUDIO_QUEUE_ROOT"]) / "runs"
        source = run_root / SOURCE
        receipt = json.loads((source / "receipt.json").read_text())
        assert receipt["revision"] == "198240fb667f2c1d2c6f5b70a9a28929d825b809" and receipt["status"] == "done"
        contrast = compare_sparse(source / "worlds.npz", counts)
        improvement = summary["inferred_minus_fixed"]["shifted"]
        pass_sparse = (contrast["shifted"]["interval"][0] >= -.02 and improvement["mean"] >= .04
                       and improvement["interval"][0] > 0 and
                       all(r["legitimate"] >= .98 for r in summary["rates"]["inferred"].values()))
        summary.update(decision="sparse_layout_sufficient" if pass_sparse else "unresolved",
                       probes_per_plane=PROBES, sampled_upper_fraction=3*PROBES/SIDE**2, sparse_minus_full=contrast)
    if args.stratified:
        random_source = run_root / "spatial-sparse-inferred-regions-884736"
        prior = json.loads((random_source / "receipt.json").read_text())
        assert prior["status"] == "done" and prior["revision"] == "8a5f1d510a943610388c95a23d57f56ab7deae1b"
        improvement = compare_sparse(random_source / "worlds.npz", counts)
        rigorous = pass_sparse and improvement["shifted"]["mean"] >= .005 and improvement["shifted"]["interval"][0] > 0
        summary.update(decision="distributed_probes_help" if rigorous else "unresolved",
                       stratified_minus_random=improvement, strata_per_axis=8)
    (out / "result.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
