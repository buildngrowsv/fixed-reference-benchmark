# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx==0.32.4", "numpy==2.5.5"]
# ///
"""Follow-up that forces the reference to carry the old board across a state reset."""
from functools import partial
from pathlib import Path
import json
import sys
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from analyze import interval
from benchmark import write_json

SEEDS=[101,211,307]
ARMS=["fixed4","writable32","fixed32"]
OUT=ROOT/"results"/"bottleneck"
EYE=mx.eye(8)


class Net(nn.Module):
    def __init__(self,arm,seed):
        super().__init__()
        self.arm=arm
        rng=np.random.default_rng(seed)
        def weights(a,b):
            return mx.array(rng.normal(0,a**-.5,(a,b)).astype("float32"))
        self.write1=weights(128,256);self.write2=weights(256,32)
        self.distract=weights(128,224)
        self.read=weights(256,256);self.work=weights(256,224)
        if arm=="writable32":self.rewrite=mx.zeros((256,32))
        self.output=weights(256,256)
        self.decode=weights(256,256)

    def __call__(self,boards):
        inputs=EYE[boards].reshape(-1,4,128)
        r=nn.silu(inputs[:,0]@self.write1)@self.write2
        if self.arm=="fixed4":
            r=mx.concatenate([r[:,:4],mx.zeros_like(r[:,4:])],axis=1)
        work=mx.zeros((boards.shape[0],224))
        for j in (1,2,3):
            work=.5*work+inputs[:,j]@self.distract
            state=nn.silu(mx.concatenate([work,r],axis=1)@self.read)
            work=work+state@self.work
            if self.arm=="writable32":r=r+state@self.rewrite
        z=mx.concatenate([work,r],axis=1)
        return ((z+nn.silu(z@self.output))@self.decode).reshape(-1,2,16,8)


def draw(rng,n):
    return rng.integers(0,8,(n,4,16),dtype=np.int32)


def main():
    mx.set_default_device(mx.gpu);OUT.mkdir(parents=True,exist_ok=True)
    check=mx.array(draw(np.random.default_rng(77),17))
    assert np.allclose(np.array(Net("fixed32",17)(check)),np.array(Net("writable32",17)(check)),atol=1e-5)
    rng=np.random.default_rng(34103);test=draw(rng,8192);q=rng.integers(16,size=len(test))
    truth=np.stack([test[np.arange(len(q)),j,q] for j in (0,3)],axis=1)
    correct={};results={}
    for arm in ARMS:
        preds=[];times=[];params=0
        for seed in SEEDS:
            model=Net(arm,seed);opt=optim.Adam(.0015)
            def obj(m,b):
                return nn.losses.cross_entropy(m(b),b[:,mx.array([0,3]),:],reduction="mean")
            grad=nn.value_and_grad(model,obj);state=[model.state,opt.state]
            @partial(mx.compile,inputs=state,outputs=state)
            def step(b):
                loss,g=grad(model,b);opt.update(model,g);return loss
            rng=np.random.default_rng(10000+seed);start=time.perf_counter()
            for _ in range(1200):
                loss=step(mx.array(draw(rng,128)));mx.eval(loss,state)
            times.append(time.perf_counter()-start)
            guesses=[]
            for i in range(0,len(q),256):
                z=np.array(model(mx.array(test[i:i+256])).argmax(-1))
                guesses.append(z[np.arange(len(z))[:,None],np.arange(2)[None,:],q[i:i+256,None]])
            pred=np.concatenate(guesses);preds.append(pred)
            params=sum(v.size for _,v in tree_flatten(model.trainable_parameters()))
            print(arm,seed,(pred==truth).mean(axis=0),flush=True);mx.clear_cache()
        z=np.stack(preds);correct[arm]=(z==truth).astype("int8")
        np.savez_compressed(OUT/(arm+".npz"),predictions=z,truth=truth)
        a=correct[arm].mean(axis=1)
        results[arm]=dict(old=float(a[:,0].mean()),current=float(a[:,1].mean()),per_seed=a.tolist(),
                          train_seconds=times,parameters=params)
    contrast={}
    for a in ARMS[:-1]:
        diff=correct["fixed32"][:,:,0]-correct[a][:,:,0]
        contrast[a]=dict(mean=float(diff.mean()),interval=interval(diff,truth[:,0]))
    write_json(OUT/"summary.json",dict(results=results,fixed32_minus=contrast,seeds=SEEDS,
        examples_seen=1200*128,test_boards=len(q),device=mx.device_info()["device_name"]))
    print(json.dumps(results),flush=True)


if __name__=="__main__":main()
