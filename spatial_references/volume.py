"""No SAVE token. Infer a write from a local sensor and an unknown adjacent sensor."""
from functools import partial
from itertools import product
import json
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from run import R, SEEDS


def neighbors(x,lib):
    return [x]+[lib.roll(x,d,axis=i) for i in (1,2,3) for d in (1,-1)]


def sample(rng,n,side,t):
    shape=(n,side,side,side)
    x=rng.choice(np.array([-1,1],np.float32),(*shape,32))
    a=rng.integers(8,size=(t,*shape),dtype=np.int32)
    sensors=rng.integers(2,size=(t,*shape,3),dtype=np.int8)
    origins=np.broadcast_to(np.arange(side**3).reshape((1,side,side,side)),shape)
    def trace(writes):
        h=mem=origins
        for action,v in zip(a,sensors):
            prev=h
            h=np.choose(action,(*neighbors(h,np),mem))
            if writes:
                fire=v[...,0].astype(bool)&np.roll(v[...,1],1,axis=3).astype(bool)
                mem=np.where(fire,prev,mem)
        return h
    source=trace(True)
    y=np.take_along_axis(x.reshape(n,-1,32),source.reshape(n,-1,1),axis=1).reshape(x.shape)
    return x,a,sensors.astype(np.float32),y,source!=trace(False)


class Cube(nn.Module):
    def __init__(self):
        super().__init__()
        self.read=mx.random.normal((8,8))*.05
        self.gate=mx.zeros((8,))
        self.bias=mx.array(-1.)

    def __call__(self,x,actions,sensors,mode):
        h=mem=x
        read=mx.softmax(self.read,axis=1)
        for a,v in zip(actions,sensors):
            prev=h
            p=read[a]
            h=sum(p[...,i:i+1]*u for i,u in enumerate((*neighbors(prev,mx),mem)))
            blue=neighbors(v[...,1],mx)
            if mode=="missing": blue[5]=v[...,2]
            elif mode=="irrelevant": blue[1]=v[...,2]
            features=mx.stack([v[...,0],*blue],axis=-1)
            g=mx.sigmoid(features@self.gate+self.bias)
            mem=mem+g[...,None]*(prev-mem)
        return h


def train(mode,seed):
    mx.random.seed(seed)
    m=Cube()
    opt=optim.Adam(.08)
    def objective(net,x,a,v,y):
        return mx.logaddexp(0.,-5.*net(x,a,v,mode)*y).mean()
    vg=nn.value_and_grad(m,objective)
    state=[m.state,opt.state]
    @partial(mx.compile,inputs=state,outputs=state)
    def step(x,a,v,y):
        loss,grad=vg(m,x,a,v,y)
        opt.update(m,grad)
        return loss
    rng=np.random.default_rng(61904+seed)
    for _ in range(500):
        x,a,v,y,_=sample(rng,12,4,8)
        loss=step(mx.array(x),mx.array(a),mx.array(v),mx.array(y))
        mx.eval(loss,state)
    return m


def main():
    for seed in SEEDS:
        stats={}
        for trained in ("full","missing"):
            m=train(trained,seed)
            weights=np.array(m.gate);bias=m.bias.item()
            truth=np.array(list(product((0,1),repeat=8)))
            hits=int(((truth@weights+bias>0)==((truth[:,0]*truth[:,6])>0)).sum())
            (R/f"cube-{trained}-{seed}.json").write_text(json.dumps(dict(read=np.array(m.read).tolist(),
                gate=weights.tolist(),bias=bias,truth_table_correct=hits),indent=2)+"\n")
            rng=np.random.default_rng(105492)
            modes=("full","missing","irrelevant") if trained=="full" else ("missing",)
            perboard={mode:[] for mode in modes}
            for _ in range(16):
                x,a,v,y,mask=sample(rng,8,6,16)
                for mode in modes:
                    pred=np.array(m(mx.array(x),mx.array(a),mx.array(v),mode))
                    bit=(pred>0)==(y>0)
                    perboard[mode].append(np.stack((mask.sum((1,2,3)),(bit.all(-1)&mask).sum((1,2,3)),
                                          (bit*mask[...,None]).sum((1,2,3,4))),axis=1))
            for name,rows in perboard.items(): stats[trained+"_"+name]=np.concatenate(rows)
        np.savez_compressed(R/f"cube-{seed}.npz",**stats)
        print(seed,{k:v.sum(0).tolist() for k,v in stats.items()},flush=True)


if __name__=="__main__": main()
