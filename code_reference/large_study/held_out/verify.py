"""Rebuild CPython panel identities and paired fixed-reference comparisons."""
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "results"
sys.path.insert(0, str(HERE.parent))
from verify import contrast


def panel(files, split, seed, count):
    entries = files[split]
    lengths = np.array([record["bytes"] - 512 for record in entries])
    bounds = np.r_[0, lengths.cumsum()]
    offsets = np.concatenate([np.random.default_rng([seed, k]).integers(bounds[-1], size=16)
                              for k in range(0, count, 16)])
    ids = np.searchsorted(bounds, offsets, side="right") - 1
    return np.column_stack((ids, offsets - bounds[ids])), np.array([entries[i]["group"] for i in ids])


def load_scores(folder, arm, name, windows, groups):
    with np.load(folder / (arm + "-" + name + ".npz")) as z:
        np.testing.assert_array_equal(z["groups"], groups)
        np.testing.assert_array_equal(z["windows"], windows)
        scores = z["scores"]
    assert scores.shape == (len(groups), 5) and np.isfinite(scores).all() and (scores >= 0).all()
    assert (scores[:, 1] <= 512).all() and (scores[:, 2] <= 512).all()
    assert (scores[:, 4] <= scores[:, 2]).all() and (scores[:, 4] <= scores[:, 1]).all()
    return scores


def read_metrics(scores):
    tokens = len(scores) * 512
    return dict(bpb=float(scores[:, 0].sum()/tokens), accuracy=float(scores[:, 1].sum()/tokens),
                repeated_identifier_bpb=float(scores[:, 3].sum()/scores[:, 2].sum()))


def main():
    folder = DATA / "large-width-rate-main"
    run = json.loads((folder / "result.json").read_text())
    receipt = json.loads((folder / "receipt.json").read_text())
    small = json.loads((DATA / "main/result.json").read_text())
    corpus = json.loads((DATA / "main/corpus.json").read_text())["files"]
    splits = {k: {r["group"] for r in rows} for k, rows in corpus.items()}
    assert not (splits["train"] & splits["val"] or splits["train"] & splits["test"] or splits["val"] & splits["test"])
    assert receipt["status"] == "done" and receipt["exit"] == 0 and receipt["job"] == "large-python-width-rate-main"
    assert receipt["revision"] == "28c77b8f9be5e1b281354e26e234d4e044d1046b"
    assert run["decision"] == "complete" and run["updates"] == small["steps"] == 16384
    assert not run["rates_matched_across_sizes"] and np.isclose(run["peak_rate"], .0004/3)
    audit = {}
    for name, split, seed, count in (("validation", "val", 881731, 1024), ("test", "test", 370915, 2048)):
        windows, groups = panel(corpus, split, seed, count)
        totals, performance = {}, {}
        for size, source, summary, parameters in (("small", DATA / "main", small["models"], 31992320),
                                                   ("large", folder, run["arms"], 284720640)):
            for arm in ("fixed", "writable"):
                values = load_scores(source, arm, name, windows, groups)
                metrics = read_metrics(values)
                assert summary[arm]["parameters"] == parameters
                np.testing.assert_allclose(metrics["bpb"], summary[arm][name + "_bpb"])
                if size == "small" and name == "test":
                    np.testing.assert_allclose(metrics["repeated_identifier_bpb"], summary[arm]["repeated_identifier_bpb"])
                    np.testing.assert_allclose(metrics["accuracy"], summary[arm]["byte_accuracy"])
                performance[size + "_" + arm] = metrics
                totals[size, arm] = values[:, 0]
        small_fixed, small_writable = totals["small", "fixed"], totals["small", "writable"]
        large_fixed, large_writable = totals["large", "fixed"], totals["large", "writable"]
        effects = dict(small=contrast(small_fixed, small_writable, groups),
                       large=contrast(large_fixed, large_writable, groups),
                       size_interaction=contrast(large_fixed-large_writable, small_fixed-small_writable, groups))
        for k, effect in effects.items():
            stated = run["contrasts"][name][k]
            np.testing.assert_equal(effect["packages"], stated["packages"])
            np.testing.assert_allclose([effect["delta"], *effect["interval"]], [stated["delta"], *stated["interval"]])
        audit[name] = dict(windows=count, packages=len(set(groups)), performance=performance, effects=effects)
    contrast_test = audit["test"]["effects"]
    large, interaction = contrast_test["large"], contrast_test["size_interaction"]
    audit["material_large_improvement"] = large["delta"] < -.02 and large["interval"][1] < 0
    audit["material_improvement_grew_with_width"] = audit["material_large_improvement"] and interaction["interval"][1] < 0
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
