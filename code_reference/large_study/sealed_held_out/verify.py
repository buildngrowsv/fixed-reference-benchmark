"""Audit the compact continuation on paired validation and reserved test windows."""
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from verify import DATA, contrast, expected_panel


def evaluate_sealed_panel(name, values, report, groups):
    metrics = {}
    for label, source in values.items():
        assert source.shape == (len(groups), 5) and np.isfinite(source).all() and (source >= 0).all()
        bits, hits, repeated, repeated_bits, repeated_hits = source.T
        assert np.all(hits <= 512) and np.all(repeated <= 512)
        assert np.all(repeated_hits <= repeated) and np.all(repeated_hits <= hits)
        metrics[label] = dict(bpb=float(bits.mean()/512), accuracy=float(hits.mean()/512),
                              repeated_identifier_bpb=float(repeated_bits.sum()/repeated.sum()))
        np.testing.assert_allclose(metrics[label]["bpb"], report["bpb"][label])
    effects = {}
    for label in ("readable", "working"):
        key = "sealed_minus_" + label
        measured = contrast(values["sealed"][:,0], values[label][:,0], groups)
        stated = report[key]
        np.testing.assert_allclose([measured["delta"], *measured["interval"]], [stated["delta"], *stated["interval"]])
        assert measured["packages"] == stated["packages"]
        effects[key] = measured
    return dict(panel=name, windows=len(groups), packages=len(set(groups)), models=metrics, contrasts=effects)


def main():
    root = DATA / "large-sealed-width-final"
    receipt = json.loads((root/"receipt.json").read_text())
    result = json.loads((root/"result.json").read_text())
    gate = json.loads((DATA/"large-sealed-width-mid/result.json").read_text())
    baseline = DATA / "large-width-rate-main"
    assert receipt["status"] == "done" and receipt["exit"] == 0
    assert receipt["revision"] == "e05f72d45368621d6ab9a445b9c3c97beaff727e"
    assert result["decision"] == "complete" and result["stage"] == "test" and result["updates"] == 16384
    assert gate["decision"] == "advance" and gate["updates"] == 4096
    assert gate["peak_rate"] == result["peak_rate"] == .0004/3
    assert result["parameters_per_compact_arm"] == 238269312
    assert result["baseline_run"] == "large-python-width-rate-main"
    metrics = {}
    panels = (("validation", "val", 881731, 1024), ("test", "test", 370915, 2048))
    for panel, split, seed, count in panels:
        windows, groups = expected_panel(split, seed, count)
        values = {}
        for label, folder, arm in (("sealed", root, "sealed"), ("working", root, "working"),
                                   ("readable", baseline, "fixed")):
            with np.load(folder/(arm+"-"+panel+".npz")) as archive:
                np.testing.assert_array_equal(archive["windows"], windows)
                np.testing.assert_array_equal(archive["groups"], groups)
                values[label] = archive["scores"]
        metrics[panel] = evaluate_sealed_panel(panel, values, result["comparisons"][panel], np.asarray(groups))
    test_windows, test_groups = expected_panel("test", 370915, 2048)
    with np.load(baseline/"writable-test.npz") as writable:
        np.testing.assert_array_equal(writable["windows"], test_windows)
        np.testing.assert_array_equal(writable["groups"], test_groups)
        metrics["exploratory_vs_wide_writable"] = {
            k: contrast(np.load(root/(k+"-test.npz"))["scores"][:,0], writable["scores"][:,0], np.asarray(test_groups))
            for k in ("sealed", "working")
        }
    held = metrics["test"]["contrasts"]
    economy = held["sealed_minus_readable"]["interval"][1] <= .02
    advantage = held["sealed_minus_working"]["delta"] <= -.02 and held["sealed_minus_working"]["interval"][1] < 0
    assert result["economical_copy"] == economy and result["early_copy_gain"] == advantage
    metrics["economical_copy"] = bool(economy)
    metrics["early_copy_gain"] = bool(advantage)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
