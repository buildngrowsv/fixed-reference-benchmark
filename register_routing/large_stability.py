"""Complete the missing original-rate arm and test a matched slower continuation."""
import sys

import numpy as np

from large_revisit import NonfiniteTrainingLoss, fit
from run import OUT, RUNS, evaluate, scores, uncertainty, write
from relay import exchange


def main():
    mode = sys.argv[1]
    if mode not in ("finish-original", "quarter-rate"):
        raise ValueError(mode)
    original_rate = mode == "finish-original"
    arms = ("rotated",) if original_rate else ("dense", "rotated")
    parent = RUNS / "large-register-dense-vs-displaced"
    results = {}
    rows = {}
    for arm in arms:
        try:
            model, _ = fit(arm, 4096, parent, None if original_rate else .0001125)
        except NonfiniteTrainingLoss as failure:
            results[arm] = dict(status="nonfinite", first_observed_update=failure.step)
            continue
        rows[arm] = evaluate(model, arm, 427361)
        np.save(OUT / (arm + "-counts.npy"), rows[arm])
        results[arm] = dict(status="complete", accuracy=scores(rows[arm]))
        if arm == "rotated" and scores(rows[arm])[0] >= .50:
            results[arm]["exchange"] = exchange(model, "long-exchange", 1, 873614, 64)
        del model
    report = dict(decision="complete", arms=results, resumed_from_update=512,
                  target=4096, learning_rate=.00045 if original_rate else .0001125)
    if set(rows) == {"dense", "rotated"}:
        report["gap"] = float(np.mean(scores(rows["rotated"])) - np.mean(scores(rows["dense"])))
        report["interval"] = uncertainty(rows["rotated"], rows["dense"])
        report["recovered_benefit"] = bool(report["gap"] >= .10 and report["interval"][0] > 0)
    write("result.json", report)


if __name__ == "__main__":
    main()
