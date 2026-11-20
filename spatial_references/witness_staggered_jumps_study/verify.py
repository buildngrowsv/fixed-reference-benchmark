"""Reconstruct scheduled jumps and endpoint contrasts from raw paired counts."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE / "results"
receipt = json.loads((ROOT / "receipt.json").read_text())
assert (receipt["job"], receipt["revision"], receipt["status"], receipt["exit"]) == (
    "spatial-witness-staggered-jumps-7077888", "2782d08b0793dc280241fdcf3dc123528de82ff0", "done", 0)
report = json.loads((ROOT / "result.json").read_text())
weights = HERE.parent / "results" / f"cube-full-{report['checkpoint']}.json"
assert sha256(weights.read_bytes()).hexdigest() == report["checkpoint_digest"]
with np.load(ROOT / "staggered.npz") as raw:
    counts, shifts = raw["counts"], raw["shifts"]
assert counts.shape == (8, 16, 3, 2, 6) and (counts[..., 2] == 192**3).all()
assert shifts.shape == (8, 16, 2, 2, 3)
assert report["worlds"] == list(range(48, 56))

sealed_offsets = []
jump_counts = []
for w, world in enumerate(report["worlds"]):
    actual = shifts[w, :, 1, 1]
    assert (shifts[w, :, 1, 0] == 0).all()
    events = np.flatnonzero(np.any(np.diff(actual, axis=0), axis=1)) + 1
    np.testing.assert_array_less(np.abs(events - [4, 8, 12]), 2)
    delta = (actual[events] - actual[events - 1]) % 96
    assert len(events) == 3 and (np.count_nonzero(delta, axis=0) == 1).all()
    assert (np.count_nonzero(delta, axis=1) == 1).all()
    assert len(set(delta[delta > 0])) == 3 and np.isin(delta[delta > 0], [28, 36, 44, 52, 60, 68]).all()
    saved = report["schedules"][w]
    assert {str(k): v.tolist() for k, v in zip(events, delta, strict=True)} == saved
    sealed_offsets.append((shifts[w, events, 0, 1] == actual[events]).tolist())
    np.testing.assert_array_equal(counts[w, :events[0], 0], counts[w, :events[0], 2])
    jump_counts.append(counts[w, events, :, 1])
jump_counts = np.stack(jump_counts)
combined = jump_counts.sum(axis=(0, 1))
retention = combined[:, 3] / combined[:, 2]
legitimate = combined[:, 4] / combined[:, 0]
recorded = report["first_round"]
np.testing.assert_allclose(retention, recorded["retention"])
np.testing.assert_allclose(legitimate, recorded["legitimate"])
draws = np.random.default_rng(976131).integers(8, size=(5000, 8))
totals = jump_counts.sum(axis=1)[draws].sum(axis=1)
rates = totals[..., 3] / totals[..., 2]
for label, arm in (("sampled_minus_stale", 0), ("sampled_minus_known", 2)):
    measured = np.percentile(rates[:, 1] - rates[:, arm], [2.5, 97.5])
    saved = recorded["changes"][label]
    np.testing.assert_allclose(measured, saved["interval"])
    np.testing.assert_allclose(retention[1] - retention[arm], saved["mean"])
delta = recorded["changes"]
success = (retention[1] - retention[0] >= .04 and delta["sampled_minus_stale"]["interval"][0] > 0
           and delta["sampled_minus_known"]["interval"][0] >= -.02 and legitimate[1] >= .98)
assert success == recorded["qualified"]
assert report["decision"] == ("sparse_router_recovers_staggered_jumps" if success else "unresolved")
np.testing.assert_array_equal(counts[:, 8:].sum(axis=(0, 1)), report["latter_half"]["counts"])
stage = jump_counts.sum(axis=0)
audit = dict(
    primary_retention=retention.tolist(),
    by_jump=(stage[..., 3] / stage[..., 2]).tolist(),
    first_round_exact_boundaries=float(np.mean(sealed_offsets)),
    false_write_fraction=(combined[:, 5] / combined[:, 2]).tolist(),
    paired_jumps=24,
)
print(json.dumps(audit, indent=2))
