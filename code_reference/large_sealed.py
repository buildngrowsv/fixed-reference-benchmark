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
    reference = np.load(baseline / "fixed-validation.npz")
    arrays = {}
    for arm in ("sealed","working"):
        make = partial(Bypass, working_exit=arm=="working", width=WIDTH, lane_width=LANE)
        model, count = fit(arm, corpus, "mid", 4096, make, peak_rate=peak_rate)
        scores, groups, windows = evaluate(model, True, corpus, "val", 1024)
        np.testing.assert_array_equal(groups,reference["groups"])
        np.testing.assert_array_equal(windows,reference["windows"])
        np.savez_compressed(OUT / (arm+"-validation.npz"), scores=scores,groups=groups,windows=windows)
        arrays[arm] = scores[:,0]
        del model
    sealed = arrays["sealed"]
    economical = grouped_difference(sealed,reference["scores"][:,0],groups)
    attribution = grouped_difference(sealed,arrays["working"],groups)
    decision = "complete"
    if args.width_rate:
        economical_copy = economical["interval"][1] <= .02
        early_copy_gain = attribution["delta"] <= -.02 and attribution["interval"][1] < 0
        decision = "advance" if economical_copy or early_copy_gain else "stop"
    write("result.json",dict(decision=decision,stage="validation",updates=4096,parameters_per_compact_arm=count,
          peak_rate=peak_rate,baseline_run=baseline_run,
          bpb={k:float(v.mean()/512) for k,v in arrays.items()},sealed_minus_readable=economical,
          sealed_minus_working=attribution))


if __name__=="__main__":
    main()
