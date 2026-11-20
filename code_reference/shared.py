"""Checkpointed equal-parameter depth-reuse experiment on real source code."""
from functools import partial
import json
import sys
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from train_code import Corpus, Block, Projection, evaluate, interval, OUT, RUNS, write, snapshot, restore

class SharedLM(nn.Module):
    def __init__(self, passes):
        super().__init__()
        self.passes=passes
        self.tokens=nn.Embedding(256,512)
        self.pos=nn.Embedding(512,512)
        self.blocks=[Block(i==2,512) for i in range(4)]
        self.norm=nn.RMSNorm(512)
        self.read=Projection(512,256,bias=False)

    def __call__(self,ids,fixed=True):
        x=(self.tokens(ids)+self.pos(mx.arange(ids.shape[1]))).astype(mx.bfloat16)
        x=self.blocks[1](self.blocks[0](x,False),False)
        for _ in range(self.passes):
            x=self.blocks[2](x,True)
        x=self.blocks[3](x,False)
        return self.read(self.norm(x).astype(x.dtype)).astype(mx.float32)

def main():
    phase=sys.argv[1]
    end=4096 if phase=="screen" else 16384
    corpus=Corpus()
    reference="cpython-reference-mid" if phase=="screen" else "cpython-reference-main"
    splits=[("val",1024)] if phase=="screen" else [("val",1024),("test",2048)]
    summary={};samples={s:{} for s,_ in splits}
    for name,passes in (("shared",7),("shallow",1)):
        mx.random.seed(101)
        model=SharedLM(passes)
        opt=optim.AdamW(lambda s:.0004*mx.minimum(1.,(s.astype(mx.float32)+1)/128),weight_decay=.03,bias_correction=True)
        dest=OUT/(name+".safetensors")
        previous=RUNS/"shared-depth-screen"/dest.name
        start=restore(model,opt,dest) if dest.exists() else restore(model,opt,previous) if phase=="main" else 0
        def objective(m,x,y):return nn.losses.cross_entropy(m(x),y,reduction="mean")
        vg=nn.value_and_grad(model,objective)
        state=[model.state,opt.state]
        @partial(mx.compile,inputs=state,outputs=state)
        def step(x,y):
            loss,g=vg(model,x,y)
            g,_=optim.clip_grad_norm(g,1.)
            opt.update(model,g)
            return loss
        begin=time.monotonic()
        for t in range(start+1,end+1):
            b,*_=corpus.batch("train",[41923,101,t],16)
            loss=step(mx.array(b[:,:-1]),mx.array(b[:,1:]))
            mx.eval(loss,state)
            if t%128==0:
                assert np.isfinite(loss.item())
                snapshot(model,opt,t,dest)
                write("progress.json",dict(arm=name,step=t,target=end,loss=loss.item(),time=time.time(),
                     tokens_per_second=(t-start)*8192/(time.monotonic()-begin)))
        summary[name]=dict(unique_parameters=sum(x.size for _,x in tree_flatten(model.parameters())),
            block_applications=3+passes,training_seconds_this_invocation=time.monotonic()-begin,start_step=start)
        for split,n in splits:
            v,groups,windows=evaluate(model,True,corpus,split,n)
            samples[split][name]=v
            fname="validation" if split=="val" else split
            np.savez_compressed(OUT/(name+"-"+fname+".npz"),scores=v,groups=groups,windows=windows)
            summary[name][split+"_bpb"]=float(v[:,0].mean()/512)
    contrasts={}
    for split,_ in splits:
        fname="validation" if split=="val" else split
        with np.load(RUNS/reference/("fixed-"+fname+".npz")) as full, np.load(OUT/("shallow-"+fname+".npz")) as shallow, np.load(OUT/("shared-"+fname+".npz")) as shared:
            for key in ("groups","windows"):
                assert np.array_equal(full[key],shallow[key]) and np.array_equal(full[key],shared[key])
            vals=samples[split]
            contrasts[split]={}
            for other,baseline in (("shallow",vals["shallow"]),("untied",full["scores"])):
                contrasts[split][other]=dict(delta=float((vals["shared"][:,0]-baseline[:,0]).mean()/512),
                                   interval=interval(vals["shared"],baseline,full["groups"]))
    v=contrasts["val"]
    go=v["shallow"]["delta"]<=-.05 and v["shallow"]["interval"][1]<0 and v["untied"]["delta"]<=.10
    write("result.json",dict(decision="advance" if phase=="screen" and go else "complete",models=summary,
        contrasts=contrasts,steps=end,seed=101,reference=reference))
if __name__=="__main__":main()

