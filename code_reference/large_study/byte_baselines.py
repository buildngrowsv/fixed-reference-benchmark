"""Fit count-based predictors on the pinned training files, score the validation windows."""
from pathlib import Path
import os
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from corpus import Corpus


def main(output):
    corpus = Corpus()
    # Unit pseudocount for every possible byte, also matching the archived unigram.
    transitions = np.ones((256, 256), dtype=np.int64)
    for _, _, stream in corpus.files["train"]:
        codes = 256*stream[:-1].astype(np.int32) + stream[1:]
        transitions += np.bincount(codes, minlength=65536).reshape(256, 256)
    nll = -np.log2(transitions / transitions.sum(1, keepdims=True))
    cost = -np.log2(corpus.unigram)
    best = transitions.argmax(1)
    scores, windows, groups = [], [], []
    for start in range(0, 1024, 16):
        data, g, ids, offsets = corpus.batch("val", [881731, start], 16)
        left, right = data[:, :-1], data[:, 1:]
        scores.extend(np.column_stack([cost[right].sum(1), (right==cost.argmin()).sum(1),
                                       nll[left, right].sum(1), (best[left]==right).sum(1)]))
        groups.extend(g)
        windows.extend(zip(ids, offsets))
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".partial.npz")
    np.savez_compressed(tmp, scores=scores, groups=groups, windows=windows)
    os.replace(tmp, output)


if __name__=="__main__":
    main(Path(sys.argv[1]))
