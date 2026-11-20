"""Independently recalculate regional routing from executed writes and choices."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE / "results"
job = json.loads((ROOT / "receipt.json").read_text())
assert job["job"] == "spatial-regional-witness-884736"
assert job["status"] == "done" and job["exit"] == 0
assert job["revision"] == "f683a4bb2e0cedfe4eb4cb82e96c2dc3e46f9bf4"
rec = json.loads((ROOT / "result.json").read_text())
for seed, digest in rec["frozen_checkpoints"].items():
    assert sha256((HERE.parent / "results" / f"cube-full-{seed}.json").read_bytes()).hexdigest() == digest
with np.load(ROOT / "worlds.npz") as data:
    counts, choices = data["counts"], data["choices"]
assert counts.shape == (3, 8, 16, 3, 2, 6)
assert choices.shape == (3, 8, 16, 4, 2, 8)
assert (counts[..., 2] == 96**3).all()
assert (counts[..., 1] <= counts[..., 0]).all() and (counts[..., 3] <= counts[..., 2]).all()
assert (counts[..., 4] <= counts[..., 0]).all() and (counts[..., 5] <= counts[..., 2] // 4).all()
assert np.isin(choices, [0, 1, 2]).all()
assert (choices[..., 0, :, :] == choices[..., 0, :, :1]).all()
assert (choices[..., 2, :, :] != choices[..., 3, :, :]).all()
assert (choices[..., 3, 0, :] == choices[..., 3, 0, :1]).all()
assert (choices[..., 3, 1, :] != choices[..., 3, 1, :1]).any()
for row in choices[..., 3, 1, :].reshape(-1, 8):
    assert sorted(np.bincount(row)) == [2, 3, 3]
by_world = counts[:, :, 8:].sum(axis=(0, 2))
totals = by_world.sum(axis=0)
np.testing.assert_array_equal(totals, rec["counts"])
aligned = (choices[:, :, 8:, :3] == choices[:, :, 8:, 3:]).mean(axis=(0, 1, 2, 5))
reads = totals[..., 3] / totals[..., 2]
writes = totals[..., 4] / totals[..., 0]
for i, name in enumerate(("global", "regional", "orientation")):
    for j, field in enumerate(("homogeneous", "heterogeneous")):
        expected = rec["rates"][name][field]
        np.testing.assert_allclose(list(expected.values()),
            [reads[i, j], writes[i, j], totals[i, j, 5] / totals[i, j, 2], aligned[i, j]])
rng = np.random.default_rng(789865)
paired = []
for draw in rng.integers(len(by_world), size=(4000, len(by_world))):
    subset = by_world[draw].sum(axis=0)
    rate = subset[..., 3] / subset[..., 2]
    paired.append(rate[1] - rate[0])
ci = np.percentile(paired, [2.5, 97.5], axis=0)
difference = reads[1] - reads[0]
for j, field in enumerate(("homogeneous", "heterogeneous")):
    result = rec["regional_minus_global"][field]
    np.testing.assert_allclose([result["mean"], *result["interval"]], [difference[j], *ci[:, j]])
passed = difference[1] >= .05 and ci[0, 1] > 0 and difference[0] >= -.02 and writes[1].min() >= .98
assert rec["decision"] == ("regional_beats_global" if passed else "unresolved")
print(json.dumps(rec, indent=2))
