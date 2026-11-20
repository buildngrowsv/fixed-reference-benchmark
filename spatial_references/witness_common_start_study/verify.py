"""Audit the common saved-word bank intervention against the original trajectories."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P = HERE / "results"
baseline = HERE.parent / "witness_staggered_jumps_study/results"
receipt = json.loads((P / "receipt.json").read_text())
assert (receipt["job"], receipt["revision"], receipt["status"], receipt["exit"]) == (
    "spatial-witness-common-start-7077888", "06b0d79fea62aad422522880344f9cbfa6b1d25b", "done", 0)
res = json.loads((P / "result.json").read_text())
seed = HERE.parent / "results/cube-full-101.json"
assert sha256(seed.read_bytes()).hexdigest() == res["checkpoint_digest"]
assert sha256((baseline / "staggered.npz").read_bytes()).hexdigest() == res["baseline_sha256"]
with np.load(P / "common-start.npz") as raw, np.load(baseline / "staggered.npz") as old:
    counts, widths = raw["counts"], raw["shifts"]
    original = old["counts"]
    np.testing.assert_array_equal(widths, old["shifts"])
assert counts.shape == original.shape == (8, 16, 3, 2, 6)
assert widths.shape == (8, 16, 2, 2, 3)
np.testing.assert_array_equal(counts[:, :, 2], original[:, :, 2])
np.testing.assert_array_equal(counts[..., [0, 2, 4, 5]], original[..., [0, 2, 4, 5]])
events = [np.flatnonzero((np.diff(world[:, 1, 1], axis=0) != 0).any(axis=1)) + 1 for world in widths]
assert [len(e) for e in events] == [3] * 8
panel = np.stack([[arm[i, event, :, 1] for i, event in enumerate(events)] for arm in (original, counts)])
totals = panel.sum(axis=(1, 2))
rates = totals[..., 3] / totals[..., 2]
np.testing.assert_allclose(rates, res["retention"])
per_jump = panel.sum(axis=1)
by_jump = per_jump[..., 3]/per_jump[..., 2]
np.testing.assert_allclose(by_jump, res["by_jump"])
gains = rates[:, 1] - rates[:, 0]
np.testing.assert_allclose(gains, res["observed_sampled_minus_stale"])
draws = np.random.default_rng(708919).integers(8, size=(5000, 8))
def gap(history):
    totals = history[draws].sum(axis=(1, 2))
    rates = totals[..., 3]/totals[..., 2]
    return rates[:, 1] - rates[:, 0]
effect = gains[0] - gains[1]
lo, hi = np.percentile(gap(panel[0]) - gap(panel[1]), [2.5, 97.5])
np.testing.assert_allclose([effect, lo, hi],
                           [res["gain_change_from_history"]["mean"], *res["gain_change_from_history"]["interval"]])
if effect >= .01 and lo > 0:
    decision = "retention_history_widens_gain"
elif effect <= -.01 and hi < 0:
    decision = "retention_history_narrows_gain"
elif lo >= -.01 and hi <= .01:
    decision = "gain_robust_to_common_start"
else:
    decision = "unresolved"
assert res["decision"] == decision
print(json.dumps(dict(decision=decision, retention=rates.tolist(), gains=gains.tolist(),
                      gains_by_jump=(by_jump[..., 1] - by_jump[..., 0]).tolist(),
                      same_offsets_and_write_counts=True,
                      history_effect=dict(mean=float(effect), interval=[float(lo), float(hi)])), indent=2))
