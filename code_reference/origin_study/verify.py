"""Reconstruct the two-candidate readout and exchange counts."""
import json
from pathlib import Path
import random
import string
import numpy as np

ROOT = Path(__file__).resolve().parents[1]/"results/large-reference-origin"


def main():
    result = json.loads((ROOT/"result.json").read_text())
    receipt = json.loads((ROOT/"receipt.json").read_text())
    assert receipt["status"]=="done" and receipt["revision"]=="23bec78b27f28ec4b762d2d173e5e2c54792a5ff"
    expected = []
    for i in range(18000,18512):
        rng = random.Random(f"cpython-ref-swap-20261119-{i}")
        rng.choices(string.ascii_lowercase,k=5)
        expected.append(list(map(ord,rng.sample(string.ascii_lowercase,2))))
    with np.load(ROOT/"margins.npz") as z:
        np.testing.assert_array_equal(z["gold"],expected)
        metrics = {k:z[k] for k in z.files if k!="gold"}
    assert all(x.shape==(512,2) and np.isfinite(x).all() for x in metrics.values())
    np.testing.assert_array_equal(metrics["whole_state"],-metrics["native"][:,::-1])
    clean = (metrics["native"]>0).all(1)
    assert int(clean.sum())==result["clean"]
    names = ("reference","neighbor","whole_site","whole_state")
    switched = {k:clean & (metrics[k]<0).all(1) for k in names}
    assert {k:int(v.sum()) for k,v in switched.items()}==result["switches"]
    for k,count in result["readout_correct"].items():
        assert int((metrics["readout_"+k]>0).all(1).sum())==count
    with np.load(ROOT/"readouts.npz") as f:
        for k in ("reference","neighbor"):
            assert f[k+"_weight"].shape==(192,26) and f[k+"_intercept"].shape==(26,)
    diff = switched["reference"].astype(int)-switched["neighbor"]
    sample = np.random.default_rng(59472).choice(len(diff),(4000,len(diff)))
    ci = np.quantile(diff[sample].mean(1),[.025,.975])
    np.testing.assert_allclose([diff.mean(),*ci],[result["fixed_minus_neighbor"],*result["interval"]])
    qualifies = clean.mean()>=.5 and diff.mean()>=.05 and ci[0]>0
    assert result["decision"]==("advance" if qualifies else "complete")
    diagnosis = result["readout_correct"]["reference"]>=.8*len(clean) and switched["whole_site"].mean()>=.2 and switched["reference"].mean()<.05
    assert bool(diagnosis)==result["readable_low_use"]
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
