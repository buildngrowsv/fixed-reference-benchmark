"""Calculate paired board/seed effects from saved per-board counts; check counterfactuals."""
import json
from pathlib import Path
import numpy as np

R=Path(__file__).resolve().parent/"results"
SEEDS=(101,211,307)


def load(pattern,key):
    rows=[]
    for s in SEEDS:
        with np.load(R/(pattern.format(s=s)+".npz")) as d: rows.append(d[key])
    return np.stack(rows)


def score(arr):
    n,e,b=arr.sum(axis=(0,1))
    assert np.array_equal(arr[0,:,0],arr[1,:,0]) and np.array_equal(arr[1,:,0],arr[2,:,0])
    assert ((arr[:,:,1]<=arr[:,:,0]) & (arr[:,:,2]>=32*arr[:,:,1]) & (arr[:,:,2]<=32*arr[:,:,0])).all()
    return dict(sites=int(arr[0,:,0].sum()),exact=e/n,bit=b/(32*n),per_seed=(arr[:,:,1].sum(1)/arr[:,:,0].sum(1)).tolist())


def paired(a,b):
    assert np.array_equal(a[:,:,0],b[:,:,0])
    rng=np.random.default_rng(338817)
    out=[]
    n=a.shape[1]
    for _ in range(40):
        i=rng.integers(n,size=(125,n))
        j=rng.integers(3,size=(125,3))
        x=(a[:,:,1]-b[:,:,1])[j[:,:,None],i[:,None,:]].sum((1,2))
        denom=3*a[0,:,0][i].sum(1)
        out.extend((x/denom).tolist())
    return [float((a[:,:,1]-b[:,:,1]).sum()/a[:,:,0].sum()),*np.quantile(out,[.025,.975]).tolist()]


def main():
    data={"primary":{},"fresh":{},"snapshot":{}}
    for shape in ("training_shape","long","wide"):
        data["primary"][shape]={}
        for arm in ("fixed","writable","late"):
            data["primary"][shape][arm]={k:score(load(arm+"-{s}",shape+"_"+k)) for k in ("normal","all","swapped","hard","copy")}
    for k in ("fixed","writable","convex","writes_off","door_open","door_closed","door_unrestricted_on_open"):
        data["fresh"][k]=score(load("intervention-{s}",k))
    for seed in SEEDS:
        with np.load(R/f"intervention-{seed}.npz") as f:
            assert np.array_equal(f["door_changed"],f["door_open"][:,0])
            assert np.array_equal(f["door_different_predictions"],f["door_changed"])
    for depth in (8,24):
        data["snapshot"][str(depth)]={}
        for cond in ("fixed_normal","event_normal","event_off","event_no_save"):
            arm,mode=cond.split("_",1)
            data["snapshot"][str(depth)][cond]=score(load("snapshot-{s}",f"{arm}_{depth}_{mode}"))
    data["paired"]={
        "primary_fixed_minus_writable":paired(load("fixed-{s}","long_normal"),load("writable-{s}","long_normal")),
        "fresh_fixed_minus_convex":paired(load("intervention-{s}","fixed"),load("intervention-{s}","convex")),
        "event_gates_on_minus_off":paired(load("snapshot-{s}","event_24_normal"),load("snapshot-{s}","event_24_no_save"))}
    (R/"summary.json").write_text(json.dumps(data,indent=2)+"\n")
    print(data["paired"])
    for k in data["fresh"]: print(k,data["fresh"][k])
    for s in SEEDS: print("write gates",s,json.loads((R/f"snapshot-event-{s}.json").read_text())["write"])


if __name__=="__main__": main()
