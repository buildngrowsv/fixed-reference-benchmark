"""Reconstruct the replay contrasts in each architecture from saved predictions."""
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rehearsal_study"))
from verify import ROOT, external


def file_contrast(a, b):
    *project_means, mean, lo, hi = external(a, b)
    return dict(projects=project_means, delta=mean, interval=[lo, hi])


def package_contrast(a, b, groups):
    delta = a - b
    ids = np.unique(groups)
    sums = np.asarray([delta[groups == key].sum() for key in ids])
    sizes = np.asarray([(groups == key).sum() for key in ids])
    draws = np.random.default_rng(9651).choice(len(ids), (5000, len(ids)))
    boot = sums[draws].sum(1) / (512 * sizes[draws].sum(1))
    return dict(delta=delta.mean() / 512, interval=np.quantile(boot, [.025, .975]).tolist())


def main():
    arms = [
        ("early", "ordinary", "ecosystem/projects.npz", "sealed_after", "ecosystem/sealed-retention.npz"),
        ("early", "replay", "rehearsal/external.npz", "scores", "rehearsal/cpython.npz"),
        ("late", "ordinary", "ecosystem-working/projects.npz", "after", "ecosystem-working/working-retention.npz"),
        ("late", "replay", "working-rehearsal/external.npz", "scores", "working-rehearsal/cpython.npz"),
    ]
    files, packages, levels = {}, {}, {}
    items = groups = windows = None
    for arch, mixture, external_path, key, cpython_path in arms:
        x = np.load(ROOT / external_path)[key]
        with np.load(ROOT / cpython_path) as z:
            y = z["scores"][:, 0]
            if items is None:
                items, groups, windows = x[:, :3], z["groups"], z["windows"]
            np.testing.assert_array_equal(x[:, :3], items)
            np.testing.assert_array_equal(z["groups"], groups)
            np.testing.assert_array_equal(z["windows"], windows)
        files[arch, mixture] = x
        packages[arch, mixture] = y
        levels[arch + "_" + mixture] = dict(external=x[:, 3].mean(), cpython=y.mean() / 512)
    assert len(items) == 3072 and len(groups) == 2048 and len(np.unique(groups)) == 49
    comparisons = {
        arch: dict(external=file_contrast(files[arch, "replay"], files[arch, "ordinary"]),
                   cpython=package_contrast(packages[arch, "replay"], packages[arch, "ordinary"], groups))
        for arch in ("early", "late")
    }
    baseline = files["late", "ordinary"].copy()
    baseline[:, 3] += files["early", "replay"][:, 3] - files["early", "ordinary"][:, 3]
    comparisons["interaction"] = dict(
        external=file_contrast(files["late", "replay"], baseline),
        cpython=package_contrast(packages["late", "replay"] - packages["late", "ordinary"],
                                 packages["early", "replay"] - packages["early", "ordinary"], groups),
    )
    receipt = json.loads((ROOT / "working-rehearsal/result.json").read_text())
    for contrast, suffix in (("late", ""), ("interaction", "_interaction")):
        for domain, estimate in comparisons[contrast].items():
            expected = receipt[domain + suffix]
            np.testing.assert_allclose([estimate["delta"], *estimate["interval"]],
                                       [expected["macro" if domain == "external" else "delta"], *expected["interval"]])
            if domain == "external":
                np.testing.assert_allclose(estimate["projects"], expected["project_differences"])
    assert all(x["interval"][1] < 0 for x in comparisons["late"].values()) == receipt["works_without_early_copy"]
    print(json.dumps(dict(levels=levels, comparisons=comparisons), indent=2))


if __name__ == "__main__":
    main()
