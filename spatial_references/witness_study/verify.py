"""Reconstruct the geometry comparison from per-round counts and checkpoint hashes."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
source = ROOT / "results"
receipt = json.loads((source / "receipt.json").read_text())
assert receipt["revision"] == "447ae9d6834f2e3f632ba1b4a225c17f9bb546d6"
assert receipt["status"] == "done" and receipt["exit"] == 0
result = json.loads((source / "result.json").read_text())
for seed, expected in result["frozen_checkpoints"].items():
    checkpoint = ROOT.parent / "results" / f"cube-full-{seed}.json"
    assert sha256(checkpoint.read_bytes()).hexdigest() == expected
data = np.load(source / "per-world.npy")
assert data.shape == (3, 8, 16, 3, 2, 6)
assert (data[..., 2] == 64**3).all()
assert np.all(data[..., 1] <= data[..., 0]) and np.all(data[..., 3] <= data[..., 2])
assert np.all(data[..., 4] <= data[..., 0]) and np.all(data[..., 5] <= data[..., 2] // 4)
assert np.all(data[:, :, :, 0, :, 5] == 64**3 // 4)
np.testing.assert_array_equal(data[:, :, :, 1, 1], data[:, :, :, 2, 1])
worlds = data[:, :, 8:].sum(axis=(0, 2))
totals = worlds.sum(axis=0)
np.testing.assert_array_equal(totals, result["counts"])
for j, arm in enumerate(("two_ticks", "near", "far")):
    for i, stimulus in enumerate(("tiles", "columns")):
        n, a, m, b, saves, faults = totals[j, i]
        rates = (saves / n, a / n, b / m, faults / m)
        np.testing.assert_allclose(rates, list(result["rates"][arm][stimulus].values()))
rng = np.random.default_rng(727143)
estimates = []
for group in rng.integers(len(worlds), size=(4000, len(worlds))):
    subset = worlds[group].sum(axis=0)
    estimates.append(subset[2, :, 3] / subset[2, :, 2] - subset[1, :, 3] / subset[1, :, 2])
for i, contrast in enumerate(result["far_minus_near"].values()):
    np.testing.assert_allclose(contrast["interval"], np.percentile(np.array(estimates)[:, i], (2.5, 97.5)))
    np.testing.assert_allclose(contrast["mean"], totals[2, i, 3] / totals[2, i, 2] - totals[1, i, 3] / totals[1, i, 2])
assert result["decision"] == "geometry_matters"
print(json.dumps(result, indent=2))
