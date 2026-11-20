"""A physically narrower middle segment and a read-only forward bypass."""
from functools import partial
import sys
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from train_code import CodeLM, Block, Corpus, OUT, RUNS, evaluate, interval, snapshot, restore, write

class Bypass(CodeLM):
    def __init__(self, working_exit=False, magnitude_exit=False, width=512, lane_width=64):
        super().__init__(width, lane_width)
        self.working_width=width-lane_width
        self.lane_width=lane_width
        d=self.working_width
        self.working_exit=working_exit
        self.magnitude_exit=magnitude_exit
        for i in range(2,9):
            src=self.blocks[i]
            b=Block(False,d)
            qkv=mx.split(src.qkv.weight,3,axis=0)
            b.qkv.weight=mx.concatenate([w[:d,:d] for w in qkv])
            b.out.weight=src.out.weight[:d,:d]
            b.up.weight=src.up.weight[:4*d,:d]
            b.down.weight=src.down.weight[:d,:4*d]
            b.norm1.weight=src.norm1.weight[:d]
            b.norm2.weight=src.norm2.weight[:d]
            self.blocks[i]=b

    def __call__(self,ids,fixed=True):
        x=(self.tokens(ids)+self.pos(mx.arange(ids.shape[1]))).astype(mx.bfloat16)
        x=self.blocks[1](self.blocks[0](x,False),False)
        x,lane=mx.split(x,[self.working_width],axis=-1)
        for block in self.blocks[2:9]:
            x=block(x,False)
        if self.working_exit:
            lane=x[...,-self.lane_width:]
            if self.magnitude_exit:
                lane=mx.abs(lane)
        x=self.blocks[-1](mx.concatenate([x,lane],axis=-1),False)
        return self.read(self.norm(x).astype(x.dtype)).astype(mx.float32)

def main():
    phase=sys.argv[1]
    magnitude_exit=sys.argv[2:] == ["magnitude-exit"]
    working_exit=magnitude_exit or sys.argv[2:] == ["working-exit"]
    name="working-exit" if working_exit else "sealed"
    if magnitude_exit:
        name="magnitude-exit"
        assert phase=="main"
    end=4096 if phase=="screen" else 16384
    corpus=Corpus()
    mx.random.seed(101)
    model=Bypass(working_exit,magnitude_exit)
    params=sum(w.size for _,w in tree_flatten(model.parameters()))
    opt=optim.AdamW(lambda s:.0004*mx.minimum(1.,(s.astype(mx.float32)+1)/128),weight_decay=.03,bias_correction=True)
    path=OUT/"bypass.safetensors"
    start=restore(model,opt,path) if path.exists() else 0
    if not start and phase=="main" and not magnitude_exit:
        parent_id="working-exit-screen" if working_exit else "sealed-bypass-screen"
        start=restore(model,opt,RUNS/parent_id/path.name)
    def objective(m,x,y):return nn.losses.cross_entropy(m(x),y,reduction="mean")
    vg=nn.value_and_grad(model,objective)
    state=[model.state,opt.state]
    @partial(mx.compile,inputs=state,outputs=state)
    def step(x,y):
        loss,grad=vg(model,x,y)
        grad,_=optim.clip_grad_norm(grad,1.)
        opt.update(model,grad)
        return loss
    clock=time.monotonic()
    for t in range(start+1,end+1):
        b,*_=corpus.batch("train",[41923,101,t],16)
        loss=step(mx.array(b[:,:-1]),mx.array(b[:,1:]));mx.eval(loss,state)
        if t%64==0:
            if not np.isfinite(loss.item()):raise ValueError("nonfinite loss")
            snapshot(model,opt,t,path)
            write("progress.json",dict(arm=name,step=t,target=end,parameters=params,time=time.time(),
                tokens_per_second=(t-start)*8192/(time.monotonic()-clock),loss=loss.item()))
    training_seconds=time.monotonic()-clock
    parent=RUNS/("cpython-reference-mid" if phase=="screen" else "cpython-reference-main")
    metrics={}
    for split,n in ([("val",1024)] if phase=="screen" else [("val",1024),("test",2048)]):
        v,g,w=evaluate(model,True,corpus,split,n)
        file="validation" if split=="val" else "test"
        np.savez_compressed(OUT/(file+".npz"),scores=v,groups=g,windows=w)
        result=dict(bpb=float(v[:,0].mean()/512),accuracy=float(v[:,1].mean()/512))
        for arm in ("fixed","writable"):
            with np.load(parent/(arm+"-"+file+".npz")) as ref:
                np.testing.assert_array_equal(ref["windows"],w)
                np.testing.assert_array_equal(ref["groups"],g)
                result[arm]=dict(delta=float((v[:,0]-ref["scores"][:,0]).mean()/512),ci=interval(v,ref["scores"],g))
        if working_exit:
            sealed=RUNS/("sealed-bypass-"+phase)/(file+".npz")
            with np.load(sealed) as ref:
                np.testing.assert_array_equal(ref["windows"],w)
                np.testing.assert_array_equal(ref["groups"],g)
                result["sealed"]=dict(delta=float((v[:,0]-ref["scores"][:,0]).mean()/512),ci=interval(v,ref["scores"],g))
        if magnitude_exit:
            with np.load(RUNS/"working-exit-main"/(file+".npz")) as ref:
                np.testing.assert_array_equal(ref["windows"],w)
                np.testing.assert_array_equal(ref["groups"],g)
                result["duplicate"]=dict(delta=float((v[:,0]-ref["scores"][:,0]).mean()/512),ci=interval(v,ref["scores"],g))
        metrics[split]=result
    go=params<=.9*31992320 and metrics["val"]["fixed"]["ci"][1]<=.05
    if working_exit:
        go=metrics["val"]["sealed"]["ci"][1]<=.05
    write("result.json",dict(arm=name,decision="advance" if phase=="screen" and go else "complete",metrics=metrics,
        parameters=params,steps=end,start_step=start,training_seconds_this_invocation=training_seconds))
if __name__=="__main__":main()
