"""Impose a fixed forward copy on a pretrained writable model."""
import argparse

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from exchange import pairs
from train_code import CodeLM, Corpus, OUT, RUNS, write

STRATA = 16


def states(model, ids):
    x = (model.tokens(ids) + model.pos(mx.arange(ids.shape[1]))).astype(mx.bfloat16)
    return model.blocks[1](model.blocks[0](x, False), False)


def predict(model, early, keep=None, partner=None):
    x = early
    source = early if partner is None else early[partner]
    if keep is not None:
        x = mx.where(keep, source, x)
    for block in model.blocks[2:9]:
        x = block(x, False)
        if keep is not None:
            x = mx.where(keep, source, x)
    x = model.blocks[9](x, False)
    return model.read(model.norm(x).astype(x.dtype)).astype(mx.float32)


def channel_stats(model, corpus, count, length):
    width = model.tokens.weight.shape[1]
    power, drift = np.zeros(width), np.zeros(width)
    for start in range(0, count, 8):
        batch, *_ = corpus.batch("train", [761819, start], 8)
        early = states(model, mx.array(batch[:, :length]))
        base = early.astype(mx.float32)
        power += np.array(mx.mean(base * base, axis=(0, 1)))
        h = early
        for block in model.blocks[2:9]:
            h = block(h, False)
            delta = h.astype(mx.float32) - base
            drift += np.array(mx.mean(delta * delta, axis=(0, 1))) / 7
    return power / (count / 8), drift / (count / 8)


def choose(power, drift):
    assert len(power) % (STRATA * 8) == 0
    rng = np.random.default_rng(107829)
    strata = np.split(np.argsort(power, kind="stable"), STRATA)
    priority = drift / np.maximum(power, 1e-6)
    ranked, random = [], []
    for channels in strata:
        ordered = channels[np.argsort(priority[channels], kind="stable")]
        n = len(ordered) // 8
        ranked.extend(ordered[:n])
        random.extend(rng.choice(ordered[n:], size=n, replace=False))
    return np.asarray(ranked), np.asarray(random)


def sample_interval(values, rng):
    idx = rng.integers(len(values), size=(3000, len(values)))
    return np.quantile(values[idx].mean(1), [.025, .975]).tolist()


def cluster_interval(delta, groups):
    names, member = np.unique(groups, return_inverse=True)
    sums = np.bincount(member, weights=delta)
    counts = np.bincount(member)
    idx = np.random.default_rng(572219).integers(len(names), size=(4000, len(names)))
    result = sums[idx].sum(1) / counts[idx].sum(1)
    return np.quantile(result, [.025, .975]).tolist()


def validation(model, corpus, masks, count, length, seed):
    losses = {k: [] for k in ["native", *masks]}
    windows, packages = [], []
    for start in range(0, count, 8):
        rows, group, ids, offsets = corpus.batch("val", [seed, start], 8)
        x, y = mx.array(rows[:, :length]), mx.array(rows[:, 1:length + 1])
        early = states(model, x)
        for k, keep in [("native", None), *masks.items()]:
            z = predict(model, early, keep)
            if k == "native" and start == 0:
                ref = model(x, False)
                np.testing.assert_array_equal(np.array(z), np.array(ref))
            err = nn.losses.cross_entropy(z, y, reduction="none")
            losses[k].extend(np.array(mx.mean(err, axis=1)) / np.log(2.))
        windows.extend(zip(ids, offsets))
        packages.extend(group)
    return {k: np.array(v) for k, v in losses.items()}, np.array(packages), np.array(windows)


def exchange_tests(model, masks, n, first):
    cases = list(pairs(n, start=first))
    margins = {name: [] for name in ["native", *masks, "selected_swap", "random_swap"]}
    targets = []
    for offset in range(0, n, 8):
        batch = cases[offset:offset + 8]
        prompts = [p for programs, _ in batch for p in programs]
        y = np.asarray([a for _, pair in batch for a in pair])
        other = np.arange(len(y)) ^ 1
        lengths = np.asarray(list(map(len, prompts)))
        byte_ids = np.zeros((len(prompts), lengths.max()), dtype=np.int32)
        for i, prompt in enumerate(prompts):
            byte_ids[i, :lengths[i]] = np.frombuffer(prompt.encode("ascii"), dtype=np.uint8)
        h = states(model, mx.array(byte_ids))
        partner = mx.array(other)
        modes = [("native", None, None)] + [(k, keep, None) for k, keep in masks.items()]
        modes += [(k + "_swap", keep, partner) for k, keep in masks.items()]
        for name, keep, source in modes:
            values = np.array(predict(model, h, keep, source)[mx.arange(len(y)), mx.array(lengths - 1)])
            positions = np.arange(len(y))
            margins[name].extend((values[positions, y] - values[positions, y[other]]).reshape(-1, 2))
        targets.extend(y.reshape(-1, 2))
    return {k: np.asarray(v) for k, v in margins.items()}, np.asarray(targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        mx.set_default_device(mx.cpu)
    else:
        mx.set_cache_limit(3 * 1024**3)
    source = "cpython-reference-mid" if args.preflight else "large-python-width-rate-mid"
    width, lane = (512, 64) if args.preflight else (1536, 192)
    length = 128 if args.preflight else 512
    model = CodeLM(width, lane)
    checkpoint = mx.load(str(RUNS / source / "writable.safetensors"))
    assert int(checkpoint["step"].item()) == 4096
    model.load_weights([(k[8:], v) for k, v in checkpoint.items() if k.startswith("weights.")])
    del checkpoint
    corpus = Corpus()
    power, drift = channel_stats(model, corpus, 16 if args.preflight else 128, length)
    selected, random = choose(power, drift)
    assert len(selected) == len(random) == lane and not np.intersect1d(selected, random).size
    masks = {k: mx.array(np.isin(np.arange(width), v)) for k, v in (("selected", selected), ("random", random))}
    seed = 493210 if args.preflight else 584134
    losses, names, windows = validation(model, corpus, masks, 16 if args.preflight else 512, length, seed)
    margins, targets = exchange_tests(model, masks, 8 if args.preflight else 512, 7168 if args.preflight else 8192)
    comparisons = {k: losses[a] - losses[b] for k, a, b in [
        ("selected_minus_native", "selected", "native"), ("random_minus_selected", "random", "selected")]}
    validation_report = {k: dict(delta=float(v.mean()), interval=cluster_interval(v, names))
                         for k, v in comparisons.items()}
    clean = {k: (margins[k] > 0).all(1) for k in ("native", "selected", "random")}
    eligible = clean["selected"] & clean["random"]
    flips = {k: (margins[k + "_swap"][eligible] < 0).all(1) for k in masks}
    delta = flips["selected"].astype(int) - flips["random"]
    bounds = sample_interval(delta, np.random.default_rng(941293)) if len(delta) else None
    advantage = float(delta.mean()) if len(delta) else None
    gate = validation_report["selected_minus_native"]["interval"][1] <= .03
    gate &= validation_report["random_minus_selected"]["interval"][0] > 0
    passed = gate and len(delta) >= 100 and advantage >= .05 and bounds[0] > 0
    np.savez_compressed(OUT / "evidence.npz", power=power, drift=drift, selected=selected, random=random,
                        groups=names, windows=windows, targets=targets,
                        **{"loss_" + k: v for k, v in losses.items()}, **{"margin_" + k: v for k, v in margins.items()})
    write("result.json", dict(decision="preflight" if args.preflight else "advance" if passed else "stop",
          checkpoint=source, width=width, lane=lane, tokens_per_window=length, validation_seed=seed,
          validation_windows=len(windows), validation_bpb={k: float(v.mean()) for k, v in losses.items()},
          contrasts=validation_report, clean_pairs={k: int(v.sum()) for k, v in clean.items()},
          test_pairs=len(targets), eligible_pairs=len(delta), target_switches={k: int(v.sum()) for k, v in flips.items()},
          switch_advantage=advantage, switch_interval=bounds))


if __name__ == "__main__":
    main()
