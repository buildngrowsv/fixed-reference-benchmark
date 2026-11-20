"""Reconstruct external sample identities and both retention contrasts."""
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from verify import DATA, contrast, expected_panel


def file_bootstrap(a,b):
    rng = np.random.default_rng(282471)
    draws,means = [],[]
    for j in range(6):
        rows = a[:,0]==j
        file = a[rows,1]
        delta = a[rows,3]-b[rows,3]
        ids = np.unique(file)
        sums = np.array([delta[file==i].sum() for i in ids])
        totals = np.array([(file==i).sum() for i in ids])
        idx = rng.choice(len(ids),(3000,len(ids)))
        draws.append(sums[idx].sum(1)/totals[idx].sum(1))
        means.append(float(delta.mean()))
    return dict(macro=float(np.mean(means)),project_differences=means,
                interval=np.quantile(np.mean(draws,axis=0),[.025,.975]).tolist())


def verified(a,b,key):
    np.testing.assert_allclose([a[key],*a["interval"]],[b[key],*b["interval"]],atol=1e-9)


def main():
    folder = DATA/"large-sealed-retention"
    summary = json.loads((folder/"result.json").read_text())
    receipt = json.loads((folder/"receipt.json").read_text())
    assert receipt["status"]=="done" and receipt["revision"]=="9e09a37cf000f8d9d8297781394bef90cbb19ef7"
    assert summary["steps"]==2048 and summary["peak_rate"]==.0004/3
    files = json.loads((folder/"files.json").read_text())
    assert files==json.loads((DATA/"ecosystem/files.json").read_text())
    names = files["projects"]["test"]
    expected = []
    for j,name in enumerate(names):
        widths = np.array([r["bytes"]-512 for r in files["files"] if r["project"]==name])
        edges = np.r_[0,widths.cumsum()]
        for i in range(32):
            point = np.random.default_rng([541982,j,i]).integers(edges[-1],size=16)
            ix = np.searchsorted(edges,point,side="right")-1
            expected.extend(zip(np.full(16,j),ix,point-edges[ix]))
    windows,groups = expected_panel("test",370915,2048)
    groups = np.asarray(groups)
    effects,forgetting,external = {},{},{}
    with np.load(folder/"projects.npz") as archive:
        for k,values in archive.items():
            assert values.shape==(3072,5) and np.isfinite(values).all()
            np.testing.assert_array_equal(values[:,:3],expected)
            external[k] = values
    for arm in ("sealed","working"):
        scores = {}
        files_to_check = ((DATA/"large-sealed-width-final"/(arm+"-test.npz"),"before"),
                          (folder/(arm+"-retention.npz"),"after"))
        for file,phase in files_to_check:
            with np.load(file) as z:
                np.testing.assert_array_equal(z["windows"],windows)
                np.testing.assert_array_equal(z["groups"],groups)
                scores[phase] = z["scores"][:,0]
            assert len(scores[phase])==2048 and np.isfinite(scores[phase]).all()
        before,after = scores["before"],scores["after"]
        forgetting[arm] = after-before
        response = summary["arms"][arm]
        scores = dict(cpython_before=float(before.mean()/512),cpython_after=float(after.mean()/512),
                      external_before=float(external[arm+"_before"][:,3].mean()),
                      external_after=float(external[arm+"_after"][:,3].mean()))
        for k,v in scores.items():
            np.testing.assert_allclose(v,response[k])
        change = contrast(after,before,groups)
        verified(change,response["cpython_change"],"delta")
        effects[arm] = dict(**scores,cpython_change=change,
                            external_change=file_bootstrap(external[arm+"_after"],external[arm+"_before"]))
    interaction = contrast(forgetting["sealed"],forgetting["working"],groups)
    verified(interaction,summary["retention_interaction"],"delta")
    between = {k:file_bootstrap(external["sealed_"+k],external["working_"+k]) for k in ("before","after")}
    for k,v in between.items():
        verified(v,summary["external_contrasts"][k],"macro")
        np.testing.assert_allclose(v["project_differences"],summary["external_contrasts"][k]["project_differences"])
    passes = interaction["delta"]<=-.03 and interaction["interval"][1]<0 and between["after"]["interval"][1]<=.02
    assert summary["decision"]==("advance" if passes else "complete")
    print(json.dumps(dict(projects=names,arms=effects,retention_interaction=interaction,external_contrasts=between,
                          registered_advantage=bool(passes)),indent=2))


if __name__=="__main__":
    main()
