"""Recreate external test windows from file lengths; reaggregate all paired losses."""
from pathlib import Path
import json
import sys
import numpy as np

def check(folder):
    root=Path(__file__).resolve().parent/"results"
    corpus=json.loads((folder/"files.json").read_text())
    summary=json.loads((folder/"result.json").read_text())
    records=corpus["files"];projects=corpus["projects"]
    assert len(set(i["sha256"] for i in records))==len(records)
    assert not (set(projects["adapt"])&set(projects["test"]))
    assert projects["test"]==summary["projects"]
    arrays=np.load(folder/"projects.npz")
    expected=[]
    for proj,name in enumerate(projects["test"]):
        files=[r for r in records if r["project"]==name]
        lengths=np.array([r["bytes"]-512 for r in files])
        ends=np.cumsum(lengths);starts=ends-lengths
        for batch in range(32):
            rng=np.random.default_rng([541982,proj,batch])
            for point in rng.integers(ends[-1],size=16):
                i=int(np.searchsorted(ends,point,side="right"));offset=int(point-starts[i])
                assert 0<=offset<lengths[i]
                expected.append([proj,i,offset])
    for name,s in arrays.items():
        np.testing.assert_array_equal(s[:,:3],expected)
        assert np.all(np.isfinite(s)) and np.all(s[:,3]>=0) and np.all((s[:,4]>=0)&(s[:,4]<=1))
        np.testing.assert_allclose(s[:,3].mean(),summary["bpb"][name])
    results={}
    for phase in ("before","after"):
        a,b=arrays["sealed_"+phase],arrays["readable_"+phase]
        rng=np.random.default_rng(282471);draws=[];means=[]
        for j,name in enumerate(projects["test"]):
            idx=a[:,0]==j;groups=a[idx,1].astype(int);delta=a[idx,3]-b[idx,3]
            cluster=np.unique(groups)
            sums=np.array([delta[groups==k].sum() for k in cluster])
            counts=np.array([(groups==k).sum() for k in cluster])
            weights=rng.choice(len(cluster),size=(3000,len(cluster)),replace=True)
            draws.append(sums[weights].sum(1)/counts[weights].sum(1))
            means.append(float(delta.mean()))
        ci=np.quantile(np.array(draws).mean(0),[.025,.975])
        s=summary["contrasts"][phase]
        np.testing.assert_allclose([*means,np.mean(means),*ci],[*s["project_differences"],s["macro"],*s["interval"]])
        results[phase]=dict(projects=dict(zip(projects["test"],means)),macro=float(np.mean(means)),interval=ci.tolist())
    for arm,oldpath in (("sealed",root/"bypass-main/test.npz"),("readable",root/"main/fixed-test.npz")):
        old=np.load(oldpath);new=np.load(folder/(arm+"-retention.npz"))
        for k in ("groups","windows"):np.testing.assert_array_equal(old[k],new[k])
        np.testing.assert_allclose((new["scores"][:,0]-old["scores"][:,0]).sum()/(512*2048),summary["cpython_loss_change"][arm])
    decision=results["after"]["interval"][1]<.02 and sum(x<.02 for x in results["after"]["projects"].values())>=5
    assert bool(decision)==summary["within_margin"]
    return dict(eligible_files=len(records),reproduced_windows=len(expected),separate_projects=results,
                within_margin=bool(decision),cpython_loss_change=summary["cpython_loss_change"])

if __name__=="__main__":print(json.dumps(check(Path(sys.argv[1])),indent=2))
