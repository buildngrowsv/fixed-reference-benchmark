"""A depth-indexed cellular network with learned local reads and optional source writes."""
import argparse
import json
from functools import partial
from pathlib import Path
from time import perf_counter

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

R=Path(__file__).resolve().parent/"results"
ARMS=("fixed","writable","late")
SEEDS=(101,211,307)
SHAPES={"training_shape":(6,8,8101),"long":(6,24,8111),"wide":(10,8,8117)}


def ports(x,ref,lib):
    return (x,lib.roll(x,1,axis=1),lib.roll(x,-1,axis=1),lib.roll(x,1,axis=2),lib.roll(x,-1,axis=2),ref)


def wall_ports(x,ref,lib,door):
    bank=list(ports(x,ref,lib))
    side=x.shape[2]
    for port,blocked,opening in ((3,(0,side//2),side//2),(4,(side-1,side//2-1),side//2-1)):
        mask=np.zeros((side,side),dtype=bool)
        mask[:,list(blocked)]=True
        if door:
            mask[side//2,opening]=False
        m=lib.array(mask)[None,...]
        if x.ndim==4: m=m[...,None]
        bank[port]=lib.where(m,x,bank[port])
    return bank


def examples(rng,n,side,depth,door=None):
    x=rng.choice(np.array([-1,1],dtype=np.float32),(n,side,side,32))
    actions=rng.integers(6,size=(depth,n,side,side),dtype=np.int32)
    start=np.broadcast_to(np.arange(side*side,dtype=np.int32).reshape(1,side,side),(n,side,side))
    def trace(mid_restore):
        h=start.copy()
        for t,a in enumerate(actions):
            ref=start if mid_restore or t==depth-1 else h
            bank=ports(h,ref,np) if door is None else wall_ports(h,ref,np,door)
            h=np.choose(a,bank)
        return h
    source=trace(True)
    causal=(source!=trace(False)) & (actions[-1]!=5)
    y=np.take_along_axis(x.reshape(n,-1,32),source.reshape(n,-1,1),axis=1).reshape(x.shape)
    return x,actions,y,causal


class Grid(nn.Module):
    def __init__(self,arm):
        super().__init__()
        self.arm=arm
        self.read=mx.random.normal((6,6))*.05
        if arm in ("writable","convex"):
            self.write=mx.zeros((6,))

    def __call__(self,x,actions,hard=False,ref=None,door=None):
        h=x
        ref=x if ref is None else ref
        for t in range(len(actions)):
            a=actions[t]
            scores=self.read
            if self.arm=="late" and t<len(actions)-1:
                scores=scores+mx.array([0,0,0,0,0,-1e6])
            weights=mx.eye(6)[scores.argmax(-1)] if hard else mx.softmax(scores,axis=-1)
            w=weights[a]
            values=ports(h,ref,mx) if door is None else wall_ports(h,ref,mx,door)
            h=sum(w[...,j:j+1]*v for j,v in enumerate(values))
            if self.arm in ("writable","convex"):
                gate=self.write[a]
                if self.arm=="writable": gate=mx.tanh(gate)
                ref=ref+gate[...,None]*(h-ref)
        return h


def fit(arm,seed,steps):
    mx.random.seed(seed)
    net=Grid(arm)
    opt=optim.Adam(.065)
    rng=np.random.default_rng(12000+seed)
    def objective(m,x,a,y):
        return mx.logaddexp(0.,-5.*m(x,a)*y).mean()
    vg=nn.value_and_grad(net,objective)
    state=[net.state,opt.state]
    @partial(mx.compile,inputs=state,outputs=state)
    def step(x,a,y):
        loss,grads=vg(net,x,a,y)
        opt.update(net,grads)
        if arm=="convex": net.write=mx.clip(net.write,0.,1.)
        return loss
    history=[]
    for k in range(1,steps+1):
        x,a,y,_=examples(rng,24,6,8)
        loss=step(mx.array(x),mx.array(a),mx.array(y))
        mx.eval(loss,state)
        if k in (1,20,100,steps):
            history.append([k,float(loss.item())])
    return net,history


def count(pred,y,mask):
    bits=(pred>0)==(y>0)
    return np.stack([mask.sum((1,2)),(bits.all(-1)&mask).sum((1,2)),
                     (bits*mask[...,None]).sum((1,2,3))],axis=1)


def evaluate(net,side,depth,seed,batches=16):
    rng=np.random.default_rng(seed)
    counts={k:[] for k in ("normal","hard","swapped","copy","all")}
    for _ in range(batches):
        x,a,y,mask=examples(rng,16,side,depth)
        xx,aa=mx.array(x),mx.array(a)
        soft=np.array(net(xx,aa))
        variants={"normal":soft,"hard":np.array(net(xx,aa,hard=True)),
                  "swapped":np.array(net(xx,aa,ref=mx.array(np.roll(x,1,axis=0)))),"copy":x,"all":soft}
        for name,p in variants.items():
            counts[name].append(count(p,y,np.ones_like(mask) if name=="all" else mask))
    return {k:np.concatenate(v) for k,v in counts.items()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    R.mkdir(exist_ok=True)
    if args.smoke:
        net,_=fit("writable",101,2)
        print(evaluate(net,3,3,212)["normal"].shape)
        return
    for seed in SEEDS:
        for arm in ARMS:
            base=R/f"{arm}-{seed}"
            if base.with_suffix(".json").is_file() and base.with_suffix(".npz").is_file(): continue
            started=perf_counter()
            net,history=fit(arm,seed,400)
            seconds=perf_counter()-started
            result={}
            for label,(side,depth,s) in SHAPES.items():
                for mode,v in evaluate(net,side,depth,s).items(): result[label+"_"+mode]=v
            np.savez_compressed(base.with_suffix(".npz"),**result)
            rec=dict(seed=seed,arm=arm,loss=history,train_seconds=seconds,read=np.array(net.read).tolist(),
                     write=np.array(net.write).tolist() if arm=="writable" else None,device=mx.device_info()["device_name"])
            base.with_suffix(".json").write_text(json.dumps(rec,indent=2)+"\n")
            print(seed,arm,round(seconds,2),"seconds",result["long_normal"].sum(0).tolist(),flush=True)


if __name__=="__main__":
    main()
