"""Relink individual answers to the independent-generator labels and training records."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/"reference_lm"))
from programs import batch

def balanced(z):
    return (z[...,1]/z[...,0]+z[...,4]/z[...,3])/2

def audit():
    folder=ROOT/"results"
    c=np.load(folder/"head-surgery/test.npz")
    report=json.loads((folder/"head-surgery/result.json").read_text())
    q=np.load(folder/"attribution/queries.npy")
    attr=json.loads((folder/"attribution/result.json").read_text())["counts"]
    gold=[]
    for i in range(32):
        _,y,op,changed,_=batch([482731,i])
        p,e=np.where((op>=2)&changed)
        gold.extend(np.column_stack((i*32+p,e,op[p,e],y[p,e])))
    np.testing.assert_array_equal(q[:,:4],gold)
    on=q[:,5]==q[:,3];native=q[:,4]==q[:,3];off=q[:,6]==q[:,3]
    for typ,key in ((2,"original"),(3,"current")):
        m=q[:,2]==typ; new=m&on&~native
        numbers=[m.sum(),new.sum(),(new&~off).sum(),(m&~on&off).sum(),(m&~on&native).sum()]
        assert numbers==[attr[key][k] for k in ("total","graft_new","new_lost_when_removed","gained_when_removed","lost_relative_to_native")]
        for arm,hit in (("native",native),("graft",on),("donor_removed",off)):
            j=1 if typ==2 else 4
            np.testing.assert_array_equal(np.bincount(q[m&hit,0],minlength=1024),c[arm][:,j])
            np.testing.assert_array_equal(np.bincount(q[m,0],minlength=1024),c[arm][:,j-1])
    for arm in c.files:
        total=c[arm].sum(0)
        np.testing.assert_allclose(total[[1,4]]/total[[0,3]],report["accuracy"][arm])
    for arm,t in report["effects"].items():
        rng=np.random.default_rng(192841);sample=[]
        for _ in range(80):
            ids=rng.choice(1024,(50,1024),replace=True)
            sample.extend(balanced(c["graft"][ids].sum(1))-balanced(c[arm][ids].sum(1)))
        np.testing.assert_allclose(np.quantile(sample,[.025,.975]),t["interval"])
        np.testing.assert_allclose(balanced(c["graft"].sum(0))-balanced(c[arm].sum(0)),t["delta"])
    independent=folder/"interchange-repeat"
    r=json.loads((independent/"result.json").read_text())["arms"]["stable"]
    labels,clean,init,control=np.load(independent/"stable.npy").transpose(1,0,2)
    targets=[]
    for i in range(128):
        a,b=batch([621789,i],32,True)
        targets.extend(np.stack([a[1][::2,-1],b[1][::2,-1]],axis=1))
    np.testing.assert_array_equal(labels,targets)
    good=(labels==clean).all(1)
    outcomes=[good&(z==labels[:,::-1]).all(1) for z in (init,control)]
    d=outcomes[0].astype(float)-outcomes[1]
    rng=np.random.default_rng(90561)
    bs=np.concatenate([d[rng.choice(len(d),(50,len(d)))].mean(1) for _ in range(100)])
    assert (len(d),good.sum(),outcomes[0].sum(),outcomes[1].sum())==tuple(r[k] for k in ("pairs","clean","double_switch_init","double_switch_write"))
    np.testing.assert_allclose([d.mean(),*np.quantile(bs,[.025,.975])],[r["delta"],*r["interval"]])
    return dict(accuracy={k:float(balanced(c[k].sum(0))) for k in c.files},effects=report["effects"],attribution=attr,
                queries=len(q),new=int((on&~native).sum()),dependent_new=int((on&~native&~off).sum()),removed_rescues=int((~on&off).sum()),replicated_exchange=r)

if __name__=="__main__":
    print(json.dumps(audit(),indent=2))
