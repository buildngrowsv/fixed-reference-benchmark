"""Reconstruct the paired interventions and the independent training result."""
import json
from pathlib import Path
import sys

import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "reference_lm"))
from programs import batch


def read(path):
    return json.loads(path.read_text())


def balanced(totals):
    return (totals[...,1]/totals[...,0] + totals[...,4]/totals[...,3])/2


def audit():
    base = HERE.parent/"results"
    records = read(base/"interchange/result.json")
    targets = []
    for i in range(64):
        a,b = batch([152843,i],32,True)
        assert np.all((a[0] != b[0]).sum(axis=1) == 1)
        targets.extend(np.stack((a[1][::2,-1], b[1][::2,-1]),axis=1))
    gold = np.array(targets)
    output = {}
    for arm,r in records["arms"].items():
        labels,clean,swap,ctrl = np.load(base/"interchange"/(arm+".npy")).transpose(1,0,2)
        np.testing.assert_array_equal(labels,gold)
        good = (clean == gold).all(1)
        effects = [good & (a == gold[:,::-1]).all(1) for a in (swap,ctrl)]
        values = [len(gold), good.sum(), *[k.sum() for k in effects],
                  (swap==gold).all(1).sum(), (ctrl==gold).all(1).sum()]
        keys = ["pairs","clean","double_switch_init","double_switch_write","correct_after_init","correct_after_write"]
        assert values == [r[k] for k in keys]
        d = effects[0].astype(float) - effects[1]
        rng = np.random.default_rng(90561)
        draws = np.concatenate([d[rng.choice(len(d),(100,len(d)))].mean(1) for _ in range(50)])
        np.testing.assert_allclose([d.mean(),*np.quantile(draws,[.025,.975])],[r["delta"],*r["interval"]])
        output[arm] = {k:int(v) for k,v in zip(keys,values)}
    folder = base/"second-order"
    repeat = read(folder/"result.json")
    x,y = (np.load(folder/"test.npz")[k] for k in ("stable","random"))
    assert x.shape == y.shape == (2048,6)
    tally = []
    for i in range(64):
        _,label,op,changed,guess = batch([816239,i])
        columns = []
        for kind in (2,3):
            use = changed & (op==kind)
            columns.extend([use.sum(1), ((guess==label)&use).sum(1)])
        tally.extend(np.stack(columns,axis=1))
    for arm,counts in (("stable",x),("random",y)):
        np.testing.assert_array_equal(counts[:,[0,2,3,5]],tally)
        total = counts.sum(0)
        np.testing.assert_allclose(total[[1,4]]/total[[0,3]],repeat["accuracy"][arm])
    rng = np.random.default_rng(192841)
    draws = []
    for _ in range(80):
        subjects = rng.choice(len(x),(50,len(x)),replace=True)
        draws.extend(balanced(x[subjects].sum(1))-balanced(y[subjects].sum(1)))
    gap = balanced(x.sum(0))-balanced(y.sum(0))
    ci = np.quantile(draws,[.025,.975])
    np.testing.assert_allclose([gap,*ci],[repeat["stable_minus_random"],*repeat["interval"]])
    assert repeat["replicated"] == bool(gap>=.10 and ci[0]>0)
    output["independent_fit"] = dict(programs=len(x), original_queries=int(x[:,0].sum()),
              current_queries=int(x[:,3].sum()),stable=float(balanced(x.sum(0))),random=float(balanced(y.sum(0))),
              gap=float(gap),interval=ci.tolist())
    return output


if __name__ == "__main__":
    print(json.dumps(audit(),indent=2))
