"""Two interchangeable local state banks trained through the original spatial task."""
from functools import partial
import json
from pathlib import Path
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "results"
sys.path.insert(0, str(HERE.parent / "spatial_references"))
from volume import Cube, neighbors, sample

SEEDS = (101,211,307)
CHECKS = (0,25,100,250,500)


def sensors(v):
    return mx.stack([v[...,0], *neighbors(v[...,1],mx)], axis=-1)


class Assigned(Cube):
    def tick(self, h, mem, a, v):
        w = mx.softmax(self.read,axis=1)[a]
        out = sum(w[...,i:i+1]*x for i,x in enumerate((*neighbors(h,mx),mem)))
        g = mx.sigmoid(sensors(v)@self.gate+self.bias)
        return out, mem + g[...,None]*(h-mem)

    def output(self, h, mem):
        return h


class Banks(nn.Module):
    def __init__(self, biased=False):
        super().__init__()
        # Read table indices are command, destination bank, gate branch, source port.
        bias = np.zeros((8,2,2,14), dtype=np.float32)
        if biased:
            bias[:,0,:,8:] = -10.
            bias[:,1,:,:] = -10.
            bias[:,1,0,7] = 0.
            bias[:,1,1,0] = 0.
        self.table = mx.random.normal((8,2,2,14))*.05 + mx.array(bias)
        self.gate = mx.zeros((8,2))
        self.bias = mx.array([-1.,-1.])
        self.head = mx.array([8.,-8.]) if biased else mx.random.normal((2,))*.05

    def tick(self, h, mem, a, v):
        sources = mx.stack([*neighbors(h,mx), *neighbors(mem,mx)],axis=-2)
        weights = mx.softmax(self.table,axis=-1)[a]
        gate = mx.sigmoid(sensors(v)@self.gate+self.bias)[...,None]
        mix = weights[...,0,:] + gate*(weights[...,1,:]-weights[...,0,:])
        z = mix @ sources
        return z[...,0,:], z[...,1,:]

    def output(self, h, mem):
        w = mx.softmax(self.head)
        return w[0]*h+w[1]*mem


def rollout(net,x,actions,v,exchange,corrupt=None):
    h=mem=x
    for t,(a,sens) in enumerate(zip(actions,v)):
        if t == len(actions)//2:
            flag=exchange[:,None,None,None,None]
            h,mem=mx.where(flag,mem,h),mx.where(flag,h,mem)
            midpoint=(h,mem)
            if corrupt is not None:
                partners=mx.array(np.arange(len(x),dtype=np.int32)^1)
                if corrupt==0: h=h[partners]
                else: mem=mem[partners]
        h,mem=net.tick(h,mem,a,sens)
    return net.output(h,mem),midpoint


def labels(x,a,v):
    n,side=x.shape[:2]
    size=side**3
    start=np.broadcast_to(np.arange(size).reshape((1,side,side,side)),(n,side,side,side))
    h=mem=start
    for t,(act,s) in enumerate(zip(a,v)):
        if t==len(a)//2:
            true_memory=np.take_along_axis(x.reshape(n,size,32),mem.reshape(n,size,1),axis=1).reshape(x.shape)
            different=h!=mem
            # Rebase provenance to the midpoint. An index >= size now means snapshot.
            h,mem=start,start+size
        prev=h
        h=np.choose(act,(*neighbors(prev,np),mem))
        fire=(s[...,0]>0)&(np.roll(s[...,1],1,axis=3)>0)
        mem=np.where(fire,prev,mem)
    src=h%size
    differed=np.take_along_axis(different.reshape(n,size),src.reshape(n,size),axis=1).reshape(h.shape)
    return true_memory,different,(h>=size)&differed


def panel(seed,side,depth,batches):
    rng=np.random.default_rng(seed)
    flip=np.random.default_rng(seed+773)
    for _ in range(batches):
        x,a,v,y,mask=sample(rng,8,side,depth)
        yield (x,a,v,y,mask,flip.permutation(8)<4)


def rates(pred,target,mask):
    hit=(pred>0)==(target>0)
    return np.stack([mask.sum((1,2,3)),(mask & hit.all(-1)).sum((1,2,3)),
                     (mask[...,None]*hit).sum((1,2,3,4))],axis=-1)


def evaluate(net,mode,dataset,interventions=False):
    parts={}
    def record(k,z): parts.setdefault(k,[]).append(z)
    for x,a,v,y,mask,flip in dataset:
        ex=flip if mode=="exchange" else np.zeros_like(flip)
        args=tuple(map(mx.array,(x,a,v,ex)))
        out,half=rollout(net,*args)
        out=np.array(out)
        truth,dist,uses=labels(x,a,v)
        record("primary",rates(out,y,mask))
        for j in range(2): record(f"probe{j}",rates(np.array(half[j]),truth,dist))
        if interventions:
            record("from_memory",rates(out,y,uses))
            for j in range(2):
                z,_=rollout(net,*args,corrupt=j)
                record(f"corrupt{j}",rates(np.array(z),y,uses))
    return {k:np.concatenate(p) for k,p in parts.items()}


def accuracy(counts):
    n,h,_=counts.sum(0)
    return float(h/n)


def save_json(path,content):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(content,indent=2)+"\n")
    tmp.replace(path)


def train(mode,seed):
    mx.random.seed(seed)
    net=Assigned() if mode=="assigned" else Banks(mode=="biased")
    opt=optim.Adam(.08)
    def objective(m,x,a,v,y,flag):
        out,_=rollout(m,x,a,v,flag)
        return mx.logaddexp(0.,-5.*out*y).mean()
    vg=nn.value_and_grad(net,objective)
    state=[net.state,opt.state]
    @partial(mx.compile,inputs=state,outputs=state)
    def step(x,a,v,y,flag):
        loss,grad=vg(net,x,a,v,y,flag)
        opt.update(net,grad)
        return loss
    rng=np.random.default_rng(61904+seed)
    flip=np.random.default_rng(72702+seed)
    val=list(panel(437141,4,8,4))
    history=[]
    elapsed=0.
    if hasattr(mx,"reset_peak_memory"): mx.reset_peak_memory()
    for k in range(501):
        if k:
            t=time.perf_counter()
            x,a,v,y,_=sample(rng,12,4,8)
            flag=flip.permutation(12)<6 if mode=="exchange" else np.zeros(12,dtype=bool)
            loss=step(*map(mx.array,(x,a,v,y,flag)))
            mx.eval(loss,state)
            elapsed+=time.perf_counter()-t
        if k in CHECKS:
            values=evaluate(net,mode,val)
            history.append(dict(step=k,exact=accuracy(values["primary"]),
                                probe=[accuracy(values["probe0"]),accuracy(values["probe1"])]))
    addr=int(np.argmax(history[-1]["probe"]))
    weights={k:np.array(w).tolist() for k,w in net.parameters().items()}
    return net,dict(mode=mode,seed=seed,address=addr,validation=history,train_seconds=elapsed,
                    params=sum(np.asarray(w).size for w in weights.values()),weights=weights,
                    peak_bytes=mx.get_peak_memory(),device=mx.device_info()["device_name"])


def main():
    R.mkdir(exist_ok=True)
    pilot=R/"pilot.json"
    if not pilot.exists():
        _,study=train("biased",101)
        save_json(pilot,study)
    result=json.loads(pilot.read_text())["validation"][-1]["exact"]
    print("Identity-biased pilot validation",result,flush=True)
    if result < .9: return
    for seed in SEEDS:
        for mode in ("assigned","neutral","exchange"):
            key=f"{mode}-{seed}"
            weights=R/f"{key}.json";counts=R/f"{key}.npz"
            if weights.exists() and counts.exists(): continue
            net,study=train(mode,seed)
            out=evaluate(net,mode,panel(984731,6,16,16),True)
            np.savez_compressed(counts,**out)
            save_json(weights,study)
            print(key,"address",study["address"],"exact",accuracy(out["primary"]),"seconds",round(study["train_seconds"],2),flush=True)


if __name__=="__main__": main()
