# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.5"]
# ///
"""Read-only verification of reported scores using released labels and predictions."""
from pathlib import Path
import hashlib
import json

import numpy as np

from analyze import interval


ROOT = Path(__file__).resolve().parent


def verify(protocol, balanced):
    results = protocol / "width-64"
    summary = json.loads((results / "summary.json").read_text())
    with np.load(protocol / "data/sensitive.npz", allow_pickle=False) as subset:
        start, final = subset["start"], subset["final"]
        assert len(final) == 10000 and subset["sensitive"].all()
    with np.load(protocol / "data/test.npz", allow_pickle=False) as ordinary:
        natural_truth = np.column_stack((ordinary["start"], ordinary["final"]))
    np.testing.assert_allclose((start == final).mean(), summary["copy_start_baseline"]["restore"])
    strata = final + 4 * (start != final) if balanced else final
    if balanced:
        assert np.all(np.bincount(strata, minlength=8) == 1250)
    correctness = {}
    patch_counts = np.zeros(3, dtype=int)
    for variant, scores in summary["summary"].items():
        primary, natural = [], []
        for seed in summary["training_seeds"]:
            folder = results / f"{seed}-{variant}"
            run = json.loads((folder / "result.json").read_text())
            with np.load(folder / "predictions.npz", allow_pickle=False) as pred:
                current = pred["sensitive"][:, 1]
                hits = current == final
                primary.append(hits)
                accuracy = (pred["test"] == natural_truth).mean(axis=0)
                natural.append(accuracy)
                np.testing.assert_allclose(accuracy, [run["test"]["original"], run["test"]["current"]])
                np.testing.assert_allclose(hits.mean(), run["sensitive"]["current"])
                if variant == "fixed":
                    donor_target = (final + 1) % 4
                    eligible = hits & (pred["donor"] == donor_target)
                    counts = [eligible.sum(), (eligible & (pred["patched"] == donor_target)).sum(),
                              (eligible & (pred["patched"] == final)).sum()]
                    reference = run["sensitive"]["patch"]
                    assert counts == [reference[k] for k in ("both_correct", "toward_donor", "stayed_original")]
                    patch_counts += counts
        correctness[variant] = np.array(primary, dtype="int8")
        np.testing.assert_allclose(correctness[variant].mean(), scores["restore"])
        np.testing.assert_allclose(np.mean(natural, axis=0), [scores["original"], scores["current"]])
    for variant, comparison in summary["fixed_differences"].items():
        delta = correctness["fixed"] - correctness[variant]
        np.testing.assert_allclose(delta.mean(axis=1), comparison["seeds"])
        np.testing.assert_allclose(delta.mean(), comparison["mean"])
        np.testing.assert_allclose(interval(delta, strata), comparison["interval"])
    print(f"{'Balanced' if balanced else 'Original'}: 20 runs verified. "
          f"Reference swaps redirect {patch_counts[1]} of {patch_counts[0]} eligible predictions.")


if __name__ == "__main__":
    for file, expected in [("benchmark.py", json.loads((ROOT / "runtime.json").read_text())["script_sha256"]),
                           ("balanced.py", json.loads((ROOT / "balanced/environment.json").read_text())["script_sha256"])]:
        assert hashlib.sha256((ROOT / file).read_bytes()).hexdigest() == expected
    verify(ROOT, False)
    verify(ROOT / "balanced", True)
