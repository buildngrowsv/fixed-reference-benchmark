"""Verify adaptive spatial routing from archived world reports and selected axes."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
receipt = json.loads((RESULTS / "receipt.json").read_text())
assert receipt["revision"] == "fa50ff22dd0e799a056f713525ea85e61936393d"
assert receipt["job"] == "spatial-adaptive-witness-axis" and receipt["status"] == "done" and receipt["exit"] == 0
report = json.loads((RESULTS / "result.json").read_text())
for seed, digest in report["frozen_checkpoints"].items():
    assert sha256((HERE.parent / "results" / f"cube-full-{seed}.json").read_bytes()).hexdigest() == digest
with np.load(RESULTS / "worlds.npz") as raw:
    counts, selected = raw["counts"], raw["axes"]
assert counts.shape == (3, 12, 16, 4, 2, 6) and selected.shape == (3, 12, 16, 5, 2)
assert (counts[..., 2] == 64**3).all() and (counts[..., 5] <= 64**3 // 4).all()
assert (counts[..., 4] <= counts[..., 0]).all() and (counts[..., 3] <= counts[..., 2]).all()
assert np.isin(selected, [0, 1, 2]).all()
assert (selected[..., 3, :] != selected[..., 4, :]).all()
assert (selected[..., 4, 0] == selected[:, :, :1, 4, 0]).all()
assert (selected[:, :, 8:, 4, 0] != selected[:, :, 8:, 4, 1]).all()
assert (selected[:, :, 4:, 1, :] == selected[:, :, 4:5, 1, :]).all()

by_world = counts[:, :, 8:].sum(axis=(0, 2))
totals = by_world.sum(axis=0)
choices = selected[:, :, 8:]
np.testing.assert_array_equal(totals, report["counts"])
axes_match = (choices[..., :4, :] == choices[..., 4:, :]).sum(axis=(0, 1, 2))
outcomes = ("fixed", "calibrated", "adaptive", "oracle")
environments = ("stable", "rotated")
for i, outcome in enumerate(outcomes):
    for j, environment in enumerate(environments):
        n, _, denominator, right, true_writes, bad_writes = totals[i, j]
        verdict = report["results"][outcome][environment]
        np.testing.assert_allclose(list(verdict.values()),
                                   [right / denominator, true_writes / n, bad_writes / denominator,
                                    axes_match[i, j] / (3 * 12 * 8)])

rng = np.random.default_rng(571039)
boot = []
for draw in rng.integers(len(by_world), size=(4000, len(by_world))):
    sampled = by_world[draw].sum(axis=0)
    boot.append(sampled[..., 3] / sampled[..., 2])
rates = totals[..., 3] / totals[..., 2]
values = np.array(boot)
for i in (0, 1, 3):
    for j, environment in enumerate(environments):
        delta = report["contrasts"][outcomes[i]][environment]
        limits = np.percentile(values[:, 2, j] - values[:, i, j], [2.5, 97.5])
        np.testing.assert_allclose([delta["mean"], *delta["interval"]], [rates[2, j] - rates[i, j], *limits])
assert report["decision"] == "adapts_to_rotated_faults"
print(json.dumps(report, indent=2))
