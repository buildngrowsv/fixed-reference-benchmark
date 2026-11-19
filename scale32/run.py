# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx==0.32.4", "numpy==2.5.5"]
# ///
"""Execute the planned width-256 study on the Studio GPU."""
from functools import partial
from pathlib import Path
import itertools
import json
import sys
import time
import hashlib
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
import benchmark as b
from analyze import interval

WIDTH=256
EPOCHS=16
ARMS=["fixed4","writable32","fixed32"]
b.SLOTS,b.COLORS,b.STEPS=16,8,8
b.SWAPS=list(itertools.combinations(range(b.SLOTS),2))
b.COUNT=len(b.SWAPS)+b.SLOTS
b.LEFT=np.array([a for a,_ in b.SWAPS]+list(range(b.SLOTS)))
b.RIGHT=np.array([v for _,v in b.SWAPS]+list(range(b.SLOTS)))
OUT=ROOT/"results"
CACHE=ROOT/"cache"


def data():
    specs=[("train",50000,5513,False),("val",4096,5501,True),
           ("test",8192,5503,True),("natural",8192,5507,False)]
    seen=set()
    for split,n,seed,balanced in specs:
        rng=np.random.default_rng(seed)
        cols=[]; counts=np.zeros((2,b.COLORS),int); used=0
        while used<n:
            d=b.examples(rng,8192)
            selected=[]
            for i in (np.flatnonzero(d["sensitive"]) if balanced else range(len(d["q"]))):
                k=(int(d["final"][i]!=d["start"][i]),d["final"][i])
                if balanced and counts[k]>=n//(2*b.COLORS):
                    continue
                key=d["board"][i].astype("uint8").tobytes()+d["cmd"][i].astype("uint8").tobytes()
                if key in seen:
                    continue
                seen.add(key);counts[k]+=1; selected.append(i)
                if len(selected)+used==n: break
            if selected:
                cols.append({k:v[selected] for k,v in d.items()});used+=len(selected)
        if balanced: assert np.all(counts==n//16)
        merged={k:np.concatenate([c[k] for c in cols]) for k in cols[0]}
        np.savez_compressed(CACHE/(split+".npz"),**merged)
        print("dataset",split,used,flush=True)


def load(split):
    with np.load(CACHE/(split+".npz")) as z: return dict(z)


def setup(arm,seed):
    b.REF=4 if arm=="fixed4" else 32
    return b.BoardNet(WIDTH,"writable" if arm=="writable32" else "fixed",seed)


def full(net,x,c):
    orig=net.encode(x)
    ref=orig[:,-b.REF:]
    state=orig[:,:-b.REF] if net.variant=="fixed" else orig
    for i,block in enumerate(net.edits):
        state=net.step(state,ref,c[:,i],block)
    if net.variant=="fixed":state=mx.concatenate([state,ref],axis=1)
    return (state@net.readout+net.out_bias).reshape(-1,2,b.SLOTS,b.COLORS)


def train(arm,seed):
    net=setup(arm,seed);opt=optim.Adam(.001)
    def objective(m,x,c,old,new):
        out=full(m,x,c)
        a=nn.losses.cross_entropy(out[:,0],old,reduction="mean")
        z=nn.losses.cross_entropy(out[:,1],new,reduction="none")
        return .25*a+.75*mx.mean(z*mx.where(old!=new,1.5,.75))
    vg=nn.value_and_grad(net,objective);state=[net.state,opt.state]
    @partial(mx.compile,inputs=state,outputs=state)
    def update(x,c,old,new):
        loss,g=vg(net,x,c,old,new);opt.update(net,g);return loss
    d=load("train");val=load("val")
    truth=np.take_along_axis(d["board"],b.trace_sources(d["cmd"]),axis=1)
    assert np.array_equal(truth[np.arange(len(truth)),d["q"]],d["final"])
    x,c,old,new=b.boards(d["board"]),mx.array(d["cmd"]),mx.array(d["board"]),mx.array(truth)
    mx.eval(x,c,old,new,net.parameters(),opt.state);mx.reset_peak_memory()
    curve=[];secs=0; rng=np.random.default_rng(55000+seed)
    for ep in range(1,EPOCHS+1):
        start=time.perf_counter()
        order=mx.array(rng.permutation(len(truth)).astype("int32"))
        for pos in range(0,len(truth),250):
            ix=order[pos:pos+250];loss=update(x[ix],c[ix],old[ix],new[ix]);mx.eval(loss,state)
        secs+=time.perf_counter()-start
        if ep in (4,8,16):curve.append([ep,b.evaluate(net,val)[0]["current"]])
    name=f"{seed}-{arm}";net.save_weights(str(CACHE/(name+".safetensors")))
    b.write_json(OUT/(name+".json"),dict(seed=seed,arm=arm,validation=curve,train_seconds=secs,
        peak_mib=mx.get_peak_memory()/2**20,parameters=sum(v.size for _,v in tree_flatten(net.trainable_parameters()))))
    print(name,curve,"seconds",round(secs,2),flush=True);mx.clear_cache()


def main():
    CACHE.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True);mx.set_default_device(mx.gpu)
    if not (CACHE/"natural.npz").exists():data()
    probe=load("val");x,c,q,_=b.tensors({k:v[:12] for k,v in probe.items()})
    fixed=setup("fixed32",71);w=setup("writable32",71)
    assert np.allclose(np.array(fixed(x,c,q)),np.array(w(x,c,q)),atol=.004)
    seeds=[101,211,307]
    def run_seeds(selected):
        for seed in selected:
            for arm in np.random.default_rng(seed).permutation(ARMS):
                if not (OUT/f"{seed}-{arm}.json").exists():train(arm,seed)
    run_seeds(seeds)
    def val(arm):return np.mean([json.loads((OUT/f"{s}-{arm}.json").read_text())["validation"][-1][1] for s in seeds])
    selected_val={a:float(val(a)) for a in ARMS}
    expand=all(selected_val["fixed32"]-selected_val[a]>=.01 for a in ARMS[:-1])
    b.write_json(OUT/"decision.json",dict(validation=selected_val,expand=expand))
    if expand:seeds += [401,503];run_seeds(seeds[-2:])
    result={}; scores={};labels={}
    for split in ("natural","test"):
        d=load(split);labels[split]=d["final"]
        for arm in ARMS:
            preds=[];old_acc=[]
            for seed in seeds:
                net=setup(arm,seed);net.load_weights(str(CACHE/f"{seed}-{arm}.safetensors"))
                metrics,pred,_=b.evaluate(net,d)
                preds.append(pred);old_acc.append(metrics["original"])
            pred=np.array(preds);scores[split,arm]=(pred[:,:,1]==d["final"]).astype("int8")
            np.savez_compressed(OUT/(split+"-"+arm+".npz"),predictions=pred,final=d["final"],original=d["start"])
            result.setdefault(arm,{})[split]=dict(accuracy=float(scores[split,arm].mean()),original=float(np.mean(old_acc)),seeds=scores[split,arm].mean(axis=1).tolist())
        if split=="test": strata=d["final"]+b.COLORS*(d["start"]!=d["final"])
    differences={}
    for arm in ARMS[:-1]:
        diff=scores["test","fixed32"]-scores["test",arm]
        differences[arm]=dict(mean=float(diff.mean()),interval=interval(diff,strata))
    report=dict(width=WIDTH,slots=b.SLOTS,colors=b.COLORS,commands=b.STEPS,epochs=EPOCHS,seeds=seeds,
                device=mx.device_info()["device_name"],results=result,fixed32_minus=differences)
    b.write_json(OUT/"summary.json",report)
    print(json.dumps(report),flush=True)


if __name__=="__main__":main()
