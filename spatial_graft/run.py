"""Test a transferred wall gate as a protected-memory write decision."""
from functools import partial
from itertools import product
import json
import os
from pathlib import Path
import sys

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"hidden_door"))
from experiment import Gate
OUT=Path(os.environ["STUDIO_JOB_DIR"])
PERM=[3,0,5,2,1,4]
BITS=np.array(list(product((0,1),repeat=6)),np.float32)
def truth(x):
    return (x[...,0]==x[...,3])|((x[...,1]>0)&(x[...,4]>0))
TABLE=truth(BITS).astype(np.float32)
CORR=np.random.default_rng(7612).permutation(TABLE)

class Receiver(nn.Module):
    def __init__(self,arm):
        super().__init__()
        self.arm=arm
        if arm=="fresh":
            self.native=Gate()
        else:
            if arm=="donor":
                self.donor=Gate()
                self.donor.load_weights(str(ROOT/"hidden_door/results/both_ends-101.safetensors"))
                self.donor.freeze()
            self.adapter=nn.Linear(6,6)
            self.adapter.weight=mx.eye(6);self.adapter.bias=mx.zeros(6)
    def __call__(self,x,mode="normal"):
        if self.arm=="fresh":return self.native(x)
        y=mx.sigmoid(6*(self.adapter(x)-.5))
        if mode=="disabled":return mx.zeros_like(y[...,:1])
        if self.arm=="donor" and mode=="normal":return self.donor(y)
        rows=mx.array(BITS)
        weights=mx.prod(mx.where(rows==1,y[...,None,:],1-y[...,None,:]),axis=-1)
        table=CORR if self.arm=="corrupted" or mode=="corrupted" else TABLE
        return (weights@mx.array(table))[...,None]

def episodes(rng,n,t,pair=False):
    x=rng.integers(0,2,(n,t,6)).astype(np.float32)
    words=rng.choice(np.array([-1,1],np.float32),(n,t+1,32))
    if pair:
        x[:,-1,0]=0;x[:,-1,3]=1;x[:,-1,1]=1;x[:,-1,4]=1
    def form(bits):
        index=np.where(truth(bits),np.arange(1,t+1),0).max(1)
        return bits[...,PERM],words,words[np.arange(n),index]
    if not pair:return form(x)
    shut=x.copy();shut[:,-1,4]=0
    a,b=form(x),form(shut)
    assert np.all(np.any(a[-1]!=b[-1],axis=1))
    return a,b

def rollout(m,x,w,mode="normal"):
    gates=m(x,mode)
    h=w[:,0]
    for j in range(x.shape[1]):h=h+gates[:,j]*(w[:,j+1]-h)
    return h

def train(arm,seed,data):
    mx.random.seed(seed);m=Receiver(arm);opt=optim.Adam(.012)
    order=np.random.default_rng(4029+seed)
    def loss(m,a,w,t):return mx.logaddexp(0.,-5*t*rollout(m,a,w)).mean()
    vg=nn.value_and_grad(m,loss)
    state=[m.state,opt.state]
    @partial(mx.compile,inputs=state,outputs=state)
    def step(a,w,t):
        v,g=vg(m,a,w,t);opt.update(m,g);return v
    for _ in range(120):
        ind=order.integers(128,size=32)
        mx.eval(step(*(mx.array(v[ind]) for v in data)),state)
    return m

def evaluate(m,pairs,mode="normal"):
    return np.stack([np.all((np.array(rollout(m,mx.array(x),mx.array(w),mode))>0)==(t>0),axis=-1) for x,w,t in pairs]).all(0)

def save(name,data):
    path=OUT/name;tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(data,indent=2)+"\n");os.replace(tmp,path)

def main():
    seeds=list(map(int,sys.argv[1:]))
    result={};all_scores={}
    for seed in seeds:
        data=episodes(np.random.default_rng(seed+8117),128,8)
        models={arm:train(arm,seed,data) for arm in ("donor","fresh","corrupted","analytic")}
        per={}
        for name,n,depth,key in (("val",512,8,3142),("test",2048,16,15173)):
            pairs=episodes(np.random.default_rng(key+seed),n,depth,True)
            h={arm:evaluate(m,pairs) for arm,m in models.items()}
            for mode in ("disabled","corrupted"):h["donor_"+mode]=evaluate(models["donor"],pairs,mode)
            # Verify the paired oracle from an independent sequential write loop.
            oracle=[]
            for xp,w,t in pairs:
                x=xp[...,np.argsort(PERM)]
                mem=w[:,0].copy()
                for j in range(depth):
                    go=truth(x[:,j])
                    mem[go]=w[go,j+1]
                oracle.append((mem==t).all(1))
            h["oracle"]=np.logical_and(*oracle)
            assert h["oracle"].all()
            for arm,v in h.items():all_scores[f"{seed}_{name}_{arm}"]=v
            per[name]={arm:int(v.sum()) for arm,v in h.items()}
            per[name].update(n=n,disabled_losses=int((h["donor"]&~h["donor_disabled"]).sum()),disabled_rescues=int((~h["donor"]&h["donor_disabled"]).sum()),
                replaced_losses=int((h["donor"]&~h["donor_corrupted"]).sum()),replaced_rescues=int((~h["donor"]&h["donor_corrupted"]).sum()))
        for arm,m in models.items():m.save_weights(str(OUT/f"{arm}-{seed}.safetensors"))
        result[str(seed)]=per
    np.savez_compressed(OUT/"predictions.npz",**all_scores)
    v=result[str(seeds[0])]["val"]
    go=v["donor"]>.6*v["n"] and v["donor"]-max(v["fresh"],v["corrupted"])>=.05*v["n"]
    save("result.json",dict(decision="replicate" if go and len(seeds)==1 else "complete",seeds=result,device=mx.device_info()["device_name"]))
if __name__=="__main__":main()

