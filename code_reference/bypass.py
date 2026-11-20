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
    def __init__(self):
        super().__init__()
        for i in range(2,9):
            src=self.blocks[i]
            b=Block(False,448)
            qkv=mx.split(src.qkv.weight,3,axis=0)
            b.qkv.weight=mx.concatenate([w[:448,:448] for w in qkv])
            b.out.weight=src.out.weight[:448,:448]
            b.up.weight=src.up.weight[:1792,:448]
            b.down.weight=src.down.weight[:448,:1792]
            b.norm1.weight=src.norm1.weight[:448]
            b.norm2.weight=src.norm2.weight[:448]
            self.blocks[i]=b

    def __call__(self,ids,fixed=True):
        x=(self.tokens(ids)+self.pos(mx.arange(ids.shape[1]))).astype(mx.bfloat16)
        x=self.blocks[1](self.blocks[0](x,False),False)
        x,lane=mx.split(x,[448],axis=-1)
        for block in self.blocks[2:9]:
            x=block(x,False)
        x=self.blocks[-1](mx.concatenate([x,lane],axis=-1),False)
        return self.read(self.norm(x).astype(x.dtype)).astype(mx.float32)

def main():
    phase=sys.argv[1]
    end=4096 if phase=="screen" else 16384
    corpus=Corpus()
    mx.random.seed(101)
    model=Bypass()
    params=sum(w.size for _,w in tree_flatten(model.parameters()))
    opt=optim.AdamW(lambda s:.0004*mx.minimum(1.,(s.astype(mx.float32)+1)/128),weight_decay=.03,bias_correction=True)
    path=OUT/"bypass.safetensors"
    start=restore(model,opt,path) if path.exists() else 0
    if not start and phase=="main":start=restore(model,opt,RUNS/"sealed-bypass-screen"/path.name)
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
            write("progress.json",dict(arm="sealed",step=t,target=end,parameters=params,time=time.time(),
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
        metrics[split]=result
    go=params<=.9*31992320 and metrics["val"]["fixed"]["ci"][1]<=.05
    write("result.json",dict(decision="advance" if phase=="screen" and go else "complete",metrics=metrics,
        parameters=params,steps=end,start_step=start,training_seconds_this_invocation=training_seconds))
if __name__=="__main__":main()
