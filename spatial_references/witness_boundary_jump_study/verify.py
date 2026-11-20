"""Rebuild recovery scores and paired intervals from the archived integer tallies."""
from hashlib import sha256
import json
from pathlib import Path
import runpy

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
STATS = runpy.run_path(str(HERE.parent / "stratified_regions_study/verify.py"))
receipt = json.loads((RESULTS / "receipt.json").read_text())
assert (receipt["job"], receipt["revision"], receipt["status"], receipt["exit"]) == (
    "spatial-witness-boundary-jump-7077888", "a6f918bd3edb95906ad46cc4d2cae0aa49693129", "done", 0)
report = json.loads((RESULTS / "result.json").read_text())
weights = HERE.parent / "results" / f"cube-full-{report['checkpoint']}.json"
assert sha256(weights.read_bytes()).hexdigest() == report["checkpoint_digest"]
with np.load(RESULTS / "jump.npz") as archive:
    counts, shifts = archive["counts"], archive["shifts"]
assert counts.shape == (1, 8, 16, 3, 2, 6)
assert shifts.shape == (1, 8, 16, 2, 2, 3)
assert (counts[..., 2] == 192**3).all()
actual = shifts[0, :, :, 1]
assert (actual[:, :, 0] == 0).all()
np.testing.assert_array_equal(actual[:, 8:, 1], (actual[:, :8, 1] + 48) % 96)
np.testing.assert_array_equal(counts[:, :, :8, 0], counts[:, :, :8, 2])
epochs = counts[0].reshape(8, 2, 8, 3, 2, 6).sum(axis=(0, 2))
np.testing.assert_array_equal(epochs, report["epochs"])
np.testing.assert_array_equal(epochs[1], report["counts"])
for j, name in enumerate(("fixed", "inferred", "known")):
    for i, layout in enumerate(("aligned", "shifted")):
        total = epochs[1, j, i]
        observed = [total[3]/total[2], total[4]/total[0], total[5]/total[2]]
        saved = report["rates"][name][layout]
        np.testing.assert_allclose(observed, [saved[k] for k in ("retention", "legitimate", "false_writes")])
world = counts[0, :, 8:].sum(axis=1)
for name, reference, seed in (("inferred_minus_fixed", 0, 686132), ("sampled_minus_known", 2, 818470)):
    STATS["confirm"](report[name], *STATS["delta"](world[:, 1], world[:, reference], seed))
error = np.abs(shifts[:, :, 8:, 0] - shifts[:, :, 8:, 1])
for i, layout in enumerate(("aligned", "shifted")):
    assert (error[..., i, :] == 0).mean() == report["boundary_accuracy"][layout]["exact"]
np.testing.assert_allclose(report["sample_fraction_upper"], (3*64*192 + 4*8*1024)/192**3)
assert report["decision"] == "sparse_witness_recovers_boundary_jump"
print("Verified raw recovery scores, geometry and paired intervals for eight worlds.")
