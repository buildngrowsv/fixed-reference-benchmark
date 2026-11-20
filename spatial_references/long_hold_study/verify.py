"""Audit the two registered Studio runs without MLX or simulator imports."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CHECKPOINTS = HERE.parent / "results"
REVISIONS = {
    "results": ("spatial-4096-long-hold", "15ad0bd92a7bd0b1132aefcdfc2cbd712e030294"),
    "confirmation": ("spatial-32768-write-confirmation", "4f035fbadbedc2eefc604cda23af6fe1eafb54bf"),
}


def load(name, archive):
    base = HERE / name
    receipt = json.loads((base / "receipt.json").read_text())
    assert (receipt["job"], receipt["revision"]) == REVISIONS[name]
    assert receipt["status"] == "done" and receipt["exit"] == 0
    report = json.loads((base / "result.json").read_text())
    digests = {str(seed): sha256((CHECKPOINTS / f"cube-full-{seed}.json").read_bytes()).hexdigest()
               for seed in (101, 211, 307)}
    assert digests == report["frozen_checkpoints"]
    scores = np.load(base / archive)
    assert np.issubdtype(scores.dtype, np.integer) and (scores >= 0).all()
    return scores, report


def audit_hold():
    scores, report = load("results", "per-volume.npy")
    assert scores.shape == (3, 64, 5)
    assert np.all(scores[..., 1:] <= scores[..., :1])
    np.testing.assert_array_equal(scores[0, :, 0], scores[1, :, 0])
    np.testing.assert_array_equal(scores[0, :, 0], scores[2, :, 0])
    labels = ("soft_short", "soft_long", "hard_short", "hard_long")
    for seed, row in zip(("101", "211", "307"), scores.sum(axis=1), strict=True):
        assert report["by_seed"][seed]["eligible"] == row[0]
        assert report["by_seed"][seed]["correct"] == dict(zip(labels, row[1:].tolist(), strict=True))
    totals = scores.sum(axis=(0, 1))
    for label, score in zip(labels, totals[1:], strict=True):
        np.testing.assert_allclose(report["pooled_rates"][label], score / totals[0])
    return {"eligible": int(totals[0]), "counts": dict(zip(labels, totals[1:].tolist(), strict=True)),
            "rates": report["pooled_rates"], "decision": report["decision"]}


def audit_faults():
    scores, report = load("confirmation", "per-world.npy")
    assert scores.shape == (3, 24, 16, 2, 3, 4)
    assert np.all(scores[..., 1] <= scores[..., 0])
    assert np.all(scores[..., 3] <= scores[..., 2])
    assert np.all(scores[..., 2] == 32**3)
    np.testing.assert_array_equal(scores[0, :, :, 0, 0, 0], scores[0, :, :, 1, 2, 0])
    assert np.all(scores[..., 0] == scores[0:1, :, :, 0:1, 0:1, 0])
    totals = scores[:, :, 8:].sum(axis=(0, 1, 2))
    np.testing.assert_array_equal(totals, report["counts"])
    for i, stimulus in enumerate(("separated", "bundled")):
        for j, policy in enumerate(("single_hard", "double_soft", "double_hard")):
            sample, hit, voxels, correct = totals[i, j]
            cell = report["per_stimulus"][stimulus][policy]
            np.testing.assert_allclose([hit / sample, correct / voxels],
                                       [cell["after_valid_write"], cell["after_fault_interval"]])
    by_world = scores[:, :, 8:, 0].sum(axis=(0, 2))
    rng = np.random.default_rng(832859)
    effects = []
    for row in rng.integers(len(by_world), size=(4000, len(by_world))):
        totals_draw = by_world[row].sum(axis=0)
        effects.append(totals_draw[2, 3] / totals_draw[2, 2] - totals_draw[0, 3] / totals_draw[0, 2])
    ci = np.percentile(effects, [2.5, 97.5])
    np.testing.assert_allclose(ci, report["double_minus_single_separated"]["interval"])
    return {"counts": totals.tolist(), "rates": report["per_stimulus"],
            "contrast": report["double_minus_single_separated"], "decision": report["decision"]}


if __name__ == "__main__":
    print(json.dumps({"long_hold": audit_hold(), "timed_faults": audit_faults()}, indent=2))
