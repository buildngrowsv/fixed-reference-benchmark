"""Compare an exact early copy and a same-size late copy in wider Python models."""
from functools import partial
import sys

import numpy as np

from bypass import Bypass
from large_fixed import fit
from train_code import Corpus, OUT, RUNS, evaluate, mx, tree_flatten, write
from verify import grouped_difference

WIDTH = 1536
LANE = 192


def main():
    if sys.argv[1:] == ["smoke"]:
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
    reference = np.load(RUNS / "large-python-reference-mid/fixed-validation.npz")
    arrays = {}
    for arm in ("sealed","working"):
        make = partial(Bypass, working_exit=arm=="working", width=WIDTH, lane_width=LANE)
        model, count = fit(arm, corpus, "mid", 4096, make)
        scores, groups, windows = evaluate(model, True, corpus, "val", 1024)
        np.testing.assert_array_equal(groups,reference["groups"])
        np.testing.assert_array_equal(windows,reference["windows"])
        np.savez_compressed(OUT / (arm+"-validation.npz"), scores=scores,groups=groups,windows=windows)
        arrays[arm] = scores[:,0]
        del model
    sealed = arrays["sealed"]
    economical = grouped_difference(sealed,reference["scores"][:,0],groups)
    attribution = grouped_difference(sealed,arrays["working"],groups)
    write("result.json",dict(decision="complete",stage="validation",updates=4096,parameters_per_compact_arm=count,
          bpb={k:float(v.mean()/512) for k,v in arrays.items()},sealed_minus_readable=economical,
          sealed_minus_working=attribution))


if __name__=="__main__":
    main()
