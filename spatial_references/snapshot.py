"""A forward reference which may capture the working state at an intermediate layer."""
from functools import partial
import json
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from run import R,SEEDS,count,ports


def boards(rng,n,t):
    side=6
    x=rng.choice(np.array([-1,1],np.float32),(n,side,side,32))
    a=rng.integers(7,size=(t,n,side,side),dtype=np.int32)
    initial=np.broadcast_to(np.arange(side*side).reshape(1,side,side),(n,side,side))
    def trace(save):
        h=mem=initial
        for action in a:
            prev=h
            h=np.choose(action,(*ports(prev,mem,np),prev))
            if save: mem=np.where(action==6,prev,mem)
        return h
    source=trace(True)
    y=np.take_along_axis(x.reshape(n,-1,32),source.reshape(n,-1,1),axis=1).reshape(x.shape)
    return x,a,y,source!=trace(False)


class Snapshot(nn.Module):
    def __init__(self,enabled):
        super().__init__()
        self.read=mx.random.normal((7,6))*.05
        self.enabled=enabled
        if enabled: self.write=mx.zeros((7,))

    def __call__(self,x,actions,mode="normal"):
        h=mem=x
        w=mx.softmax(self.read,axis=1)
        for a in actions:
            prev=h
            p=w[a]
            h=sum(p[...,i:i+1]*v for i,v in enumerate(ports(prev,mem,mx)))
            if self.enabled and mode!="off":
                gate=self.write[a]
                if mode=="no_save": gate=mx.where(a==6,0.,gate)
                mem=mem+gate[...,None]*(prev-mem)
        return h


def train(enabled,seed):
    mx.random.seed(seed)
    m=Snapshot(enabled)
    opt=optim.Adam(.065)
    def loss(net,x,a,y): return mx.logaddexp(0.,-5.*net(x,a)*y).mean()
    vg=nn.value_and_grad(m,loss)
    state=[m.state,opt.state]
    @partial(mx.compile,inputs=state,outputs=state)
    def update(x,a,y):
        l,g=vg(m,x,a,y)
        opt.update(m,g)
        if enabled: m.write=mx.clip(m.write,0.,1.)
        return l
    rng=np.random.default_rng(91700+seed)
    for _ in range(400):
        x,a,y,_=boards(rng,24,8)
        l=update(mx.array(x),mx.array(a),mx.array(y))
        mx.eval(l,state)
    return m


def main():
    for seed in SEEDS:
        results={}
        for enabled,name in ((False,"fixed"),(True,"event")):
            m=train(enabled,seed)
            for depth in (8,24):
                rng=np.random.default_rng(51173+depth)
                modes=("normal","off","no_save") if enabled else ("normal",)
                cells={k:[] for k in modes}
                for _ in range(64):
                    x,a,y,mask=boards(rng,16,depth)
                    xx,aa=mx.array(x),mx.array(a)
                    for k in modes: cells[k].append(count(np.array(m(xx,aa,k)),y,mask))
                for k,items in cells.items(): results[f"{name}_{depth}_{k}"]=np.concatenate(items)
            (R/f"snapshot-{name}-{seed}.json").write_text(json.dumps(dict(
                read=np.array(m.read).tolist(),write=np.array(m.write).tolist() if enabled else None),indent=2)+"\n")
        np.savez_compressed(R/f"snapshot-{seed}.npz",**results)
        print(seed,{k:v.sum(0).tolist() for k,v in results.items()},flush=True)


if __name__=="__main__": main()
