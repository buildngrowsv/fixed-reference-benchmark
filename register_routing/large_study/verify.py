"""Reconstruct the width comparison from program counts and paired answer swaps."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = HERE.parent / "results" / "large-width"
sys.path.insert(0, str(ROOT / "reference_lm"))
from programs import batch


def balanced(counts):
    return (counts[..., 1] / counts[..., 0] + counts[..., 4] / counts[..., 3]) / 2


def main():
    original = np.load(RAW.parent / "test.npz")
    counts = np.load(RAW / "counts.npz")
    result = json.loads((RAW / "result.json").read_text())
    for arm in ("rotated", "dense"):
        np.testing.assert_array_equal(original[arm], counts["small_" + arm])
    denominators = []
    for i in range(32):
        _, gold, op, changed, guess = batch([625137, i])
        cols = []
        for kind in (2, 3):
            use = changed & (op == kind)
            cols.extend([use.sum(1), ((guess == gold) & use).sum(1)])
        denominators.extend(np.stack(cols, axis=1))
    for name, rows in counts.items():
        assert rows.shape == (1024, 6) and np.all(rows >= 0)
        assert np.all(rows[:, [1, 4]] <= rows[:, [0, 3]])
        np.testing.assert_array_equal(rows[:, [0, 2, 3, 5]], denominators)
        totals = rows.sum(0)
        np.testing.assert_allclose(totals[[1, 4]] / totals[[0, 3]], result["accuracy"][name])

    rng = np.random.default_rng(192841)
    draws = []
    for _ in range(80):
        indices = rng.integers(1024, size=(50, 1024))
        means = {name: balanced(rows[indices].sum(1)) for name, rows in counts.items()}
        small = means["small_rotated"] - means["small_dense"]
        large = means["large_rotated"] - means["large_dense"]
        draws.append(np.stack((large, large-small), axis=1))
    large_ci, interaction_ci = np.quantile(np.concatenate(draws), [.025, .975], axis=0).T
    np.testing.assert_allclose(large_ci, result["large_gap_interval"])
    np.testing.assert_allclose(interaction_ci, result["change_interval"])

    expected = []
    for i in range(64):
        a, b = batch([152843, i], 32, True)
        expected.extend(np.stack((a[1][::2, -1], b[1][::2, -1]), axis=1))
    labels, clean, source, later = np.load(RAW / "large-exchange.npy").transpose(1, 0, 2)
    np.testing.assert_array_equal(labels, expected)
    both = (clean == labels).all(1)
    switches = [int((both & (pred == labels[:, ::-1]).all(1)).sum()) for pred in (source, later)]
    assert [int(both.sum()), *switches] == [result["swap"][k] for k in ("clean", "double_switch_init", "double_switch_write")]
    sizes = [8 * (12*d*d+2*d) + (54+256+32+1)*d for d in (384, 1536)]
    assert sizes == [14293632, 227043840] and all(n == sizes[1] for n in result["parameters"].values())
    print(json.dumps(dict(parameters=sizes, programs=1024, original_queries=int(totals[0]),
        current_queries=int(totals[3]), balanced={k:float(balanced(v.sum(0))) for k,v in counts.items()},
        large_gap_interval=large_ci.tolist(), change_interval=interaction_ci.tolist(),
        clean_pairs=int(both.sum()), switches=switches), indent=2))


if __name__ == "__main__":
    main()
