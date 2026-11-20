"""Check channel assignments, held-out window identities, and intervention outcomes."""
import json
from pathlib import Path
import random
import string
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
FILES = HERE.parent / "results/main/corpus.json"


def check_panel(windows, groups, seed):
    files = json.loads(FILES.read_text())["files"]["val"]
    widths = np.array([r["bytes"] - 512 for r in files])
    ends = widths.cumsum()
    expected, names = [], []
    for start in range(0, len(windows), 8):
        locations = np.random.default_rng([seed, start]).integers(ends[-1], size=8)
        for where in locations:
            idx = int(np.searchsorted(ends, where, side="right"))
            expected.append([idx, where - (ends[idx-1] if idx else 0)])
            names.append(files[idx]["group"])
    np.testing.assert_array_equal(windows, expected)
    np.testing.assert_array_equal(groups, names)
    assert len(np.unique(windows, axis=0)) == len(windows)
    with np.load(HERE.parent / "results/large-width-rate-mid/fixed-validation.npz") as earlier:
        assert set(map(tuple, windows)).isdisjoint(map(tuple, earlier["windows"]))


def audit(folder):
    result = json.loads((folder / "result.json").read_text())
    with np.load(folder / "evidence.npz") as source:
        archive = dict(source)
    power, drift = archive["power"], archive["drift"]
    assert np.isfinite(power).all() and np.isfinite(drift).all()
    assert (power >= 0).all() and (drift >= 0).all()
    control = np.random.default_rng(107829)
    selected = sorted(range(len(power)), key=lambda i: (power[i], i))
    for rank, strata in enumerate(np.array_split(selected, 16)):
        ordered = sorted(strata, key=lambda i: drift[i] / max(power[i], 1e-6))
        n = len(ordered) // 8
        region = slice(rank*n, (rank+1)*n)
        assert archive["selected"][region].tolist() == ordered[:n]
        np.testing.assert_array_equal(archive["random"][region], control.choice(ordered[n:], n, replace=False))
    assert len(archive["selected"]) == len(archive["random"]) == result["lane"]
    groups = archive["groups"]
    check_panel(archive["windows"], groups, result["validation_seed"])
    for key, mean in result["validation_bpb"].items():
        losses = archive["loss_" + key]
        assert len(losses) == result["validation_windows"] and np.isfinite(losses).all()
        np.testing.assert_allclose(losses.mean(), mean)
    for key, a, b in (("selected_minus_native","selected","native"),("random_minus_selected","random","selected")):
        delta = archive["loss_" + a] - archive["loss_" + b]
        names = sorted(set(groups))
        buckets = [delta[groups == name] for name in names]
        indices = np.random.default_rng(572219).choice(len(names), (4000, len(names)))
        sums = np.array([v.sum() for v in buckets], dtype=np.float64)
        counts = np.array([len(v) for v in buckets])
        ci = np.quantile(sums[indices].sum(1)/counts[indices].sum(1), [.025, .975])
        recorded = result["contrasts"][key]
        np.testing.assert_allclose([delta.mean(), *ci], [recorded["delta"], *recorded["interval"]], atol=1e-7)
    first = 7168 if result["decision"] == "preflight" else 8192
    for idx, pair in enumerate(archive["targets"], start=first):
        rng = random.Random(f"cpython-ref-swap-20261119-{idx}")
        rng.choices(string.ascii_lowercase, k=5)
        np.testing.assert_array_equal(pair, list(map(ord, rng.sample(string.ascii_lowercase, 2))))
    clean = {k:(archive["margin_" + k] > 0).all(1) for k in result["clean_pairs"]}
    assert {k:int(v.sum()) for k, v in clean.items()} == result["clean_pairs"]
    eligible = clean["selected"] & clean["random"]
    assert int(eligible.sum()) == result["eligible_pairs"]
    flips = {k:(archive["margin_" + k + "_swap"][eligible] < 0).all(1) for k in ("selected","random")}
    assert {k:int(v.sum()) for k, v in flips.items()} == result["target_switches"]
    if eligible.any():
        delta = flips["selected"].astype(int) - flips["random"]
        ix = np.random.default_rng(941293).choice(len(delta), (3000, len(delta)))
        ci = np.quantile(delta[ix].mean(1), [.025,.975])
        np.testing.assert_allclose([delta.mean(),*ci], [result["switch_advantage"],*result["switch_interval"]])
    else:
        assert result["switch_advantage"] is None and result["switch_interval"] is None
    if result["decision"] != "preflight":
        passed = (result["contrasts"]["selected_minus_native"]["interval"][1] <= .03
                  and result["contrasts"]["random_minus_selected"]["interval"][0] > 0
                  and result["eligible_pairs"] >= 100 and result["switch_advantage"] >= .05
                  and result["switch_interval"][0] > 0)
        assert result["decision"] == ("advance" if passed else "stop")
    return result


if __name__ == "__main__":
    print(json.dumps(audit(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"), indent=2))
