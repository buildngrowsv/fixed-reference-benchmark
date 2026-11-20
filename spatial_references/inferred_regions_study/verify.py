"""Audit exact translated boundary recovery and downstream memory retention."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
source = ROOT / "results"
job = json.loads((source / "receipt.json").read_text())
assert (job["job"], job["revision"], job["status"], job["exit"]) == (
    "spatial-inferred-regions-884736", "198240fb667f2c1d2c6f5b70a9a28929d825b809", "done", 0)
report = json.loads((source / "result.json").read_text())
for seed, signature in report["frozen_checkpoints"].items():
    assert sha256((ROOT.parent / "results" / f"cube-full-{seed}.json").read_bytes()).hexdigest() == signature
with np.load(source / "worlds.npz") as raw:
    counts, shifts = raw["counts"], raw["shifts"]
assert counts.shape == (3, 8, 16, 3, 2, 6)
assert shifts.shape == (3, 8, 16, 2, 2, 3)
assert (counts[..., 2] == 96**3).all()
assert (counts[..., 4] <= counts[..., 0]).all() and (counts[..., 3] <= counts[..., 2]).all()
assert (counts[..., 5] <= counts[..., 2]//4).all()
guide = shifts[:, :, :, 1]
assert (guide[:, :, :, 0] == 0).all()
assert ((guide[:, :, :, 1] >= 4) & (guide[:, :, :, 1] <= 44) & (guide[:, :, :, 1] % 4 == 0)).all()
assert (guide == guide[:, :, :1]).all()
assert ((shifts >= 0) & (shifts < 48)).all()

rows = counts[:, :, 8:].sum(axis=(0, 2))
total = rows.sum(axis=0)
np.testing.assert_array_equal(total, report["counts"])
rates = total[..., 3]/total[..., 2]
accept = total[..., 4]/total[..., 0]
diff = rates[1] - rates[0]
for i, name in enumerate(("fixed", "inferred", "known")):
    for j, scenario in enumerate(("aligned", "shifted")):
        value = report["rates"][name][scenario]
        np.testing.assert_allclose(list(value.values()), [rates[i,j], accept[i,j], total[i,j,5]/total[i,j,2]])
error = np.abs(shifts[:, :, 8:, 0] - shifts[:, :, 8:, 1])
folded = np.minimum(error, 48 - error)
rng = np.random.default_rng(686132)
boot = []
for sample in rng.integers(len(rows), size=(4000,len(rows))):
    block = rows[sample].sum(axis=0)
    score = block[..., 3] / block[..., 2]
    boot.append(score[1]-score[0])
ci = np.percentile(boot, [2.5,97.5], axis=0)
for j, scenario in enumerate(("aligned", "shifted")):
    corrected = folded[...,j,:]
    assert report["boundary_accuracy"][scenario] == dict(exact=(corrected==0).mean(),
                                                          within_four=(corrected<=4).mean())
    comparison = report["inferred_minus_fixed"][scenario]
    np.testing.assert_allclose([comparison["mean"], *comparison["interval"]], [diff[j], *ci[:,j]])
assert report["decision"] == ("inferred_boundaries_help" if diff[1] >= .05 and ci[0,1] > 0
                              and diff[0] >= -.02 and accept[1].min() >= .98 else "unresolved")
print(json.dumps(report, indent=2))
