"""Post hoc readout test using saved top-ranked write positions and visible payloads."""
import json
from pathlib import Path
import sys

import numpy as np

ARMS=("donor_clean","donor_paired","scratch_clean","scratch_paired","pretrained_clean","pretrained_paired")


def bounds(hit,eligible,draws):
    good=(hit&eligible).sum(1)[draws].sum(1)
    n=eligible.sum(1)[draws].sum(1)
    return np.quantile(good/n,[.025,.975]).tolist()


def main(folder):
    report={}
    for file in sorted(folder.glob("*.npz")):
        with np.load(file) as saved:
            payloads=saved["input_payloads"]
            draws=np.random.default_rng(713317).integers(len(payloads),size=(4000,len(payloads)))
            per_fit={}
            for arm in ARMS:
                correct={}
                for edit in ("clean","current","stale"):
                    # The prediction is the visible value at the learned top-ranked write.
                    pred=np.take_along_axis(payloads,saved[arm+"_"+edit+"_source"],axis=1)
                    correct[edit]=pred==saved[edit+"_gold"]
                clean=correct["clean"]
                eligible=clean[:,:2]
                soft=saved[arm+"_clean"]==saved["clean_gold"]
                info=dict(clean=float(clean.mean()),clean_gain_over_neural=float(clean.mean()-soft.mean()),edits={})
                for edit in ("current","stale"):
                    hit=correct[edit]
                    both=(eligible&hit).sum()
                    info["edits"][edit]=dict(conditional=float(both/eligible.sum()),
                        unconditional=float(hit.mean()),both=int(both),clean_correct=int(eligible.sum()),
                        conditional_interval=bounds(hit,eligible,draws))
                per_fit[arm]=info
            report[file.stem]=per_fit
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main(Path(sys.argv[1]))
