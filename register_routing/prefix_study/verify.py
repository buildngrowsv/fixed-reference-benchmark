"""Rebuild the graft contrasts and verify all released counterfactual targets."""
import sys,json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root.parent/"reference_lm"))
from programs import batch

def mean_score(z):
    return .5*(z[...,1]/z[...,0]+z[...,4]/z[...,3])

def audit():
    data_dir=root/"results/prefix"
    fit=json.loads((data_dir/"result.json").read_text())
    groups=np.load(data_dir/"test.npz")
    old=np.load(root/"results/head-surgery/test.npz")
    for k in ("native","graft","donor_full"):
        np.testing.assert_array_equal(groups[k],old[k])
    totals={}
    for k,v in groups.items():
        np.testing.assert_array_equal(v[:,[0,2,3,5]],old["native"][:,[0,2,3,5]])
        assert v.shape==(1024,6) and np.all(v[:,[1,4]]<=v[:,[0,3]])
        a=v.sum(0);rates=a[[1,4]]/a[[0,3]]
        np.testing.assert_allclose(rates,fit["accuracy"][k])
        totals[k]=float(rates.mean())
    for k,r in fit["effects"].items():
        rng=np.random.default_rng(192841);draw=[]
        for _ in range(80):
            ix=rng.choice(1024,(50,1024),replace=True)
            draw.extend(mean_score(groups["trained_prefix"][ix].sum(1))-mean_score(groups[k][ix].sum(1)))
        np.testing.assert_allclose([totals["trained_prefix"]-totals[k],*np.quantile(draw,[.025,.975])],[r["delta"],*r["interval"]])
    okay=all(v["delta"]>=.10 and v["interval"][0]>0 for v in fit["effects"].values())
    assert okay == (fit["decision"]=="replicate")
    result=dict(balanced=totals,contrasts=fit["effects"],qualified=okay)
    if okay:
        folder=root/"results/hybrid-interchange"
        rec=json.loads((folder/"result.json").read_text())
        labels,clean,init,ctrl=np.load(folder/"hybrid.npy").transpose(1,0,2)
        expected=[]
        for i in range(128):
            a,b=batch([192640,i],32,True)
            expected.extend(np.stack([a[1][::2,-1],b[1][::2,-1]],axis=1))
        np.testing.assert_array_equal(labels,expected)
        good=(labels==clean).all(1)
        x=good&(init==labels[:,::-1]).all(1);y=good&(ctrl==labels[:,::-1]).all(1)
        assert (len(labels),good.sum(),x.sum(),y.sum())==tuple(rec[k] for k in ("pairs","clean","source","control"))
        d=x.astype(float)-y;draw=[];rng=np.random.default_rng(118231)
        for _ in range(100):draw.extend(d[rng.choice(len(d),(50,len(d)))].mean(1))
        np.testing.assert_allclose([d.mean(),*np.quantile(draw,[.025,.975])],[rec["delta"],*rec["interval"]])
        result["fresh_exchange"]=rec
    return result

if __name__=="__main__":print(json.dumps(audit(),indent=2))
