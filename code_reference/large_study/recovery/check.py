"""Recover the learning-rate and architecture effects from matched validation bytes."""
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from verify import DATA, contrast, expected_panel


def check():
    folder = DATA / "large-width-rate-mid"
    result = json.loads((folder / "result.json").read_text())
    assert result["updates"] == 4096 and np.isclose(result["peak_rate"], .0004 / 3)
    windows, names = expected_panel()
    groups = np.asarray(names)
    values, performance = {}, {}
    for label, directory in (("original", DATA / "large-original-mid"), ("recovery", folder), ("small", DATA / "mid")):
        for arm in ("fixed", "writable"):
            with np.load(directory / (arm + "-validation.npz")) as f:
                np.testing.assert_array_equal(f["windows"], windows)
                np.testing.assert_array_equal(f["groups"], groups)
                bits, correct = f["scores"].T[:2]
                assert len(bits) == 1024 and np.isfinite(bits).all() and np.all(bits >= 0)
                assert np.all((correct >= 0) & (correct <= 512))
                values[label, arm] = bits
                performance[label + "_" + arm] = dict(bpb=bits.mean() / 512, accuracy=correct.mean() / 512)
    def verified(a, b, record):
        measured = contrast(a, b, groups)
        np.testing.assert_allclose([measured["delta"], *measured["interval"]], [record["delta"], *record["interval"]])
        assert record["packages"] == measured["packages"]
        return measured
    rates = {}
    for arm in ("fixed", "writable"):
        info = result["arms"][arm]
        assert info["parameters"] == 284720640
        np.testing.assert_allclose(info["validation_bpb"], performance["recovery_" + arm]["bpb"])
        rates[arm] = verified(values["recovery", arm], values["original", arm], info["lower_minus_original_rate_bpb"])
    contrasts = result["contrasts"]["validation"]
    treatment = verified(values["recovery", "fixed"], values["recovery", "writable"], contrasts["large"])
    interaction = verified(values["recovery", "fixed"] - values["recovery", "writable"],
                           values["small", "fixed"] - values["small", "writable"], contrasts["size_interaction"])
    recipe_interaction = contrast(values["recovery", "fixed"] - values["recovery", "writable"],
                                  values["original", "fixed"] - values["original", "writable"], groups)
    passed = all(performance["recovery_" + k]["bpb"] < 3.24 for k in ("fixed", "writable"))
    assert result["decision"] == ("advance" if passed else "stop")
    return dict(windows=len(groups), packages=len(set(groups)), performance=performance,
                reduced_minus_original_rate=rates, fixed_minus_writable=treatment, size_interaction=interaction,
                change_in_fixed_minus_writable_between_rates=recipe_interaction, passed=bool(passed))


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
