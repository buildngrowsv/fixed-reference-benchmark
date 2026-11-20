"""Verify exact frame decisions and reconstruct the paired timing analysis."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
receipt = json.loads((RESULTS / "receipt.json").read_text())
assert (receipt["job"], receipt["revision"], receipt["status"], receipt["exit"]) == (
    "spatial-compiled-witness-timing", "5bfee15fe06518ce479fbc0ebc359b24fa6a93df", "done", 0)
report = json.loads((RESULTS / "result.json").read_text())
with np.load(RESULTS / "timing.npz") as archive:
    elapsed, decisions = archive["elapsed_ns"], archive["decisions"]
assert elapsed.shape == (48, 12, 3) and (elapsed > 0).all()
assert decisions.shape == (48, 3, 11)
assert (decisions == decisions[:, :1]).all()
assert ((decisions[..., :3] >= 0) & (decisions[..., :3] < 96)).all()
assert ((decisions[..., 3:] >= 0) & (decisions[..., 3:] < 3)).all()
for seed, hash_value in report["frozen_checkpoints"].items():
    assert sha256((HERE.parent / "results" / f"cube-full-{seed}.json").read_bytes()).hexdigest() == hash_value
np.testing.assert_allclose(np.median(elapsed, axis=(0, 1))/1e6, report["median_ms"])
frame = np.median(elapsed, axis=1)
gain = np.median((frame[:, :2]/frame[:, 2:]).reshape(3, 8, 2, 2), axis=(0, 2))
draws = np.random.default_rng(483417).integers(8, size=(5000, 8))
bounds = np.percentile(np.median(gain[draws], axis=1), [2.5, 97.5], axis=0).T
for i, name in enumerate(("full_materialization", "two_gathers")):
    np.testing.assert_allclose(np.median(gain[:, i]), report["speedup"][name])
    np.testing.assert_allclose(bounds[i], report["paired_intervals"][name])
assert report["decision"] == ("compiled_observer_latency_improves"
    if np.median(gain[:, 1]) >= 1.25 and bounds[1, 0] > 1 else "unresolved")
print("Verified exact decisions, duration medians and paired speed ratios.")
