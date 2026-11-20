"""Recompute spatial memory outcomes from per-volume word counts."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RESULT = HERE.parent / "results"


def main():
    raw = RESULT / "large-controller"
    report = json.loads((raw / "result.json").read_text())
    counts = np.load(raw / "counts.npz")
    audit = {}
    for arm in ("persistent","biased"):
        old = np.load(RESULT / (arm+"-101.npz"))
        values = report["arms"][arm]
        primary = counts[arm+"_primary"]
        np.testing.assert_array_equal(primary[:,0],old["primary"][:,0])
        correct = primary.sum(0)
        n = len(primary)
        idx = np.random.default_rng(829731).integers(n//2,size=(4000,n//2))
        pairs = np.stack([2*idx,2*idx+1],axis=-1).reshape(4000,n)
        big, small = [v[pairs].sum(1) for v in (primary,old["primary"])]
        ci = np.quantile(big[:,1]/big[:,0]-small[:,1]/small[:,0],[.025,.975])
        np.testing.assert_allclose(ci,values["improvement_interval"])
        bank = values["read_bank"]
        swapped = [counts[f"{arm}_corrupt{k}"].sum(0) for k in (bank,1-bank)]
        impact = swapped[1][1]/swapped[1][0]-swapped[0][1]/swapped[0][0]
        np.testing.assert_allclose([correct[1]/correct[0],impact],[values["large"],values["selective_disruption"]])
        audit[arm] = dict(words=int(correct[0]),large_correct=int(correct[1]),small_correct=int(old["primary"][:,1].sum()),
                          improvement_interval=ci.tolist(),selective_disruption=float(impact))
    print(json.dumps(audit,indent=2))


if __name__=="__main__":
    main()
