"""Compare an exact early copy and a same-size late copy in wider Python models."""
import argparse
from functools import partial
import json

import numpy as np

from bypass import Bypass
from large_fixed import fit
from train_code import Corpus, OUT, RUNS, evaluate, mx, tree_flatten, write
from verify import grouped_difference

WIDTH = 1536
LANE = 192


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", choices=["smoke"])
    parser.add_argument("--width-rate", action="store_true")
    parser.add_argument("--continue-to-test", action="store_true")
    args = parser.parse_args()
    if args.mode == "smoke":
        mx.set_default_device(mx.cpu)
        for width, lane in ((512,64),(WIDTH,LANE)):
            mx.random.seed(101)
            m = Bypass(width=width, lane_width=lane)
            x = mx.array([[1,4,5,3]])
            early = np.array(m(x))
            m.working_exit = True
            late = np.array(m(x))
            assert early.shape == (1,4,256) and np.isfinite(early).all() and np.isfinite(late).all()
            assert np.any(early != late)
            print(width,sum(v.size for _,v in tree_flatten(m.parameters())))
        return
    corpus = Corpus()
    baseline_run = "large-python-width-rate-mid" if args.width_rate else "large-python-reference-mid"
    baseline = RUNS / baseline_run
    peak_rate = .0004 / 3 if args.width_rate else .0004
    if args.width_rate:
        recovered = json.loads((baseline / "result.json").read_text())
        assert recovered["decision"] == "advance" and recovered["updates"] == 4096
        assert np.isclose(recovered["peak_rate"], peak_rate)
    resume_run = "large-sealed-width-rate-mid"
    if args.continue_to_test:
        assert args.width_rate
        screen = json.loads((RUNS / resume_run / "result.json").read_text())
        assert screen["decision"] == "advance" and screen["updates"] == 4096
        assert np.isclose(screen["peak_rate"], peak_rate)
        baseline_run = "large-python-width-rate-main"
        baseline = RUNS / baseline_run
    phase = "main" if args.continue_to_test else "mid"
    steps = 16384 if args.continue_to_test else 4096
    panels = [("validation", "val", 1024)]
    if args.continue_to_test:
        panels.append(("test", "test", 2048))
    results = {}
    for name, _, _ in panels:
        with np.load(baseline / ("fixed-"+name+".npz")) as archive:
            results[name] = {"reference": dict(archive)}
    for arm in ("sealed", "working"):
        make = partial(Bypass, working_exit=arm == "working", width=WIDTH, lane_width=LANE)
        model, count = fit(arm, corpus, phase, steps, make, peak_rate=peak_rate, mid_run=resume_run)
        for name, split, size in panels:
            scores, groups, windows = evaluate(model, True, corpus, split, size)
            reference = results[name]["reference"]
            np.testing.assert_array_equal(groups, reference["groups"])
            np.testing.assert_array_equal(windows, reference["windows"])
            np.savez_compressed(OUT / (arm+"-"+name+".npz"), scores=scores, groups=groups, windows=windows)
            results[name][arm] = scores[:, 0]
        del model
    comparisons = {}
    for name, panel in results.items():
        reference = panel["reference"]
        sealed, working = panel["sealed"], panel["working"]
        readable, groups = reference["scores"][:, 0], reference["groups"]
        comparisons[name] = dict(
            bpb={k: float(v.mean()/512) for k, v in (("sealed", sealed), ("working", working), ("readable", readable))},
            sealed_minus_readable=grouped_difference(sealed, readable, groups),
            sealed_minus_working=grouped_difference(sealed, working, groups),
        )
    key = "test" if args.continue_to_test else "validation"
    result = comparisons[key]
    economical = result["sealed_minus_readable"]
    attribution = result["sealed_minus_working"]
    economical_copy = economical["interval"][1] <= .02
    early_copy_gain = attribution["delta"] <= -.02 and attribution["interval"][1] < 0
    decision = "complete"
    if args.width_rate and not args.continue_to_test:
        decision = "advance" if economical_copy or early_copy_gain else "stop"
    write("result.json", dict(decision=decision, stage=key, updates=steps, parameters_per_compact_arm=count,
          peak_rate=peak_rate, baseline_run=baseline_run, **result, comparisons=comparisons,
          economical_copy=bool(economical_copy), early_copy_gain=bool(early_copy_gain)))


if __name__=="__main__":
    main()
