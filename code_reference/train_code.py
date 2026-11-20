"""A checkpointed paired test on package-disjoint CPython source."""
from functools import partial
import json
import os
from pathlib import Path
import sys
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np
from corpus import Corpus,DIGEST,repeats,suffix_scores
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"reference_lm"))
from train import Block,Projection,snapshot,restore,write,OUT,RUNS

class CodeLM(nn.Module):
    def __init__(self,width=512,protected_width=64):
        super().__init__()
        self.tokens=nn.Embedding(256,width);self.pos=nn.Embedding(512,width)
        self.blocks=[Block(2<=j<9,width,protected_width) for j in range(10)]
        self.norm=nn.RMSNorm(width);self.read=Projection(width,256,bias=False)
    def __call__(self,ids,fixed):
        x=(self.tokens(ids)+self.pos(mx.arange(ids.shape[1]))).astype(mx.bfloat16)
        for j,block in enumerate(self.blocks):x=block(x,fixed and 2<=j<9)
        return self.read(self.norm(x).astype(x.dtype)).astype(mx.float32)

def evaluate(model,fixed,corpus,split,n):
    data=[];groups=[];windows=[]
    for i in range(0,n,16):
        panel_seed=881731 if split=="val" else 370915
        batch,g,ids,off=corpus.batch(split,[panel_seed,i],16)
        x,y=mx.array(batch[:,:-1]),mx.array(batch[:,1:])
        z=model(x,fixed);err=np.array(nn.losses.cross_entropy(z,y,reduction="none"))/np.log(2.)
        hit=np.array(z.argmax(-1)==y);mask=np.stack([repeats(r) for r in batch])
        data.append(np.stack([err.sum(1),hit.sum(1),mask.sum(1),(err*mask).sum(1),(hit&mask).sum(1)],axis=1))
        groups.extend(g);windows.extend(zip(ids,off))
    return np.concatenate(data),np.array(groups),np.array(windows)

def baselines(corpus,split,n,with_suffix):
    unigram=suffix=0.;suffix_hits=0
    for i in range(0,n,16):
        b,*_=corpus.batch(split,[881731 if split=="val" else 370915,i],16)
        unigram-=np.log2(corpus.unigram[b[:,1:]]).sum()
        if with_suffix:
            for row in b:
                losses,hits=suffix_scores(row,corpus.unigram)
                suffix+=losses.sum();suffix_hits+=hits.sum()
    return dict(unigram_bpb=unigram/(512*n),suffix_bpb=suffix/(512*n) if with_suffix else None,
                suffix_accuracy=suffix_hits/(512*n) if with_suffix else None)

def interval(a,b,groups):
    names=np.unique(groups);diff=a[:,0]-b[:,0]
    sums=np.array([diff[groups==g].sum() for g in names]);sizes=np.array([(groups==g).sum()*512 for g in names])
    picks=np.random.default_rng(9651).integers(len(names),size=(5000,len(names)))
    return np.quantile(sums[picks].sum(1)/sizes[picks].sum(1),[.025,.975]).tolist()

def main():
    phase,order,parent=sys.argv[1],int(sys.argv[2]),sys.argv[3]
    end={"screen":256,"mid":4096,"main":16384,"replication":16384}[phase]
    n=256 if phase=="screen" else 1024
    corpus=Corpus();write("corpus.json",dict(archive=DIGEST,files=corpus.manifest()))
    pre=baselines(corpus,"val",n,False)
    models={};valid={};stats={};started=time.monotonic()
    for arm in ("fixed","writable"):
        mx.random.seed(order);model=CodeLM();models[arm]=model
        fixed=arm=="fixed"
        opt=optim.AdamW(lambda s:.0004*mx.minimum(1.,(s.astype(mx.float32)+1)/128),weight_decay=.03,bias_correction=True)
        path=OUT/(arm+".safetensors")
        if path.exists():first=restore(model,opt,path)
        elif parent!="none":first=restore(model,opt,RUNS/parent/path.name)
        else:first=0
        def objective(m,x,y):
            return nn.losses.cross_entropy(m(x,fixed),y,reduction="mean")
        vg=nn.value_and_grad(model,objective)
        state=[model.state,opt.state]
        @partial(mx.compile,inputs=state,outputs=state)
        def step(x,y):
            loss,g=vg(model,x,y);g,_=optim.clip_grad_norm(g,1.0);opt.update(model,g);return loss
        clock=time.monotonic()
        for t in range(first+1,end+1):
            b,*_=corpus.batch("train",[41923,order,t],16)
            l=step(mx.array(b[:,:-1]),mx.array(b[:,1:]));mx.eval(l,state)
            if t==1 or t%64==0:
                if not np.isfinite(l.item()):raise ValueError("nonfinite training loss")
                snapshot(model,opt,t,path)
                speed=(t-first)*8192/max(time.monotonic()-clock,.001)
                write("progress.json",dict(arm=arm,step=t,target=end,seed=order,loss=l.item(),tokens_per_second=speed,
                        time=time.time(),parameters=sum(v.size for _,v in tree_flatten(model.parameters()))))
                print(arm,t,round(l.item(),4),round(speed),"bytes/s",flush=True)
        v,g,w=evaluate(model,fixed,corpus,"val",n);valid[arm]=v
        np.savez_compressed(OUT/(arm+"-validation.npz"),scores=v,groups=g,windows=w)
        stats[arm]=dict(validation_bpb=float(v[:,0].mean()/512),parameters=sum(z.size for _,z in tree_flatten(model.parameters())))
    bpb=[s["validation_bpb"] for s in stats.values()]
    ci=interval(valid["fixed"],valid["writable"],g)
    if phase=="screen":decision="advance" if all(np.isfinite(bpb)) and max(bpb)<pre["unigram_bpb"]-.3 else "stop"
    elif phase=="mid":
        parent_result=json.loads((RUNS/parent/"result.json").read_text())
        base=np.mean([r["validation_bpb"] for r in parent_result["models"].values()])
        # Use exactly the first 256 windows for the comparison with the screen.
        now=np.mean([v[:256,0].mean()/512 for v in valid.values()])
        decision="advance" if base-now>=.1 and all(np.isfinite(bpb)) else "stop"
    else:decision="replicate" if phase=="main" and abs(bpb[0]-bpb[1])>=.02 and ci[0]*ci[1]>0 else "complete"
    write("decision.json",dict(decision=decision,validation_interval=ci,models=stats))
    test_control=None
    if phase in ("main","replication"):
        test_control=baselines(corpus,"test",2048,True)
        for arm,m in models.items():
            v,g,w=evaluate(m,arm=="fixed",corpus,"test",2048)
            np.savez_compressed(OUT/(arm+"-test.npz"),scores=v,groups=g,windows=w)
            stats[arm].update(test_bpb=float(v[:,0].mean()/512),byte_accuracy=float(v[:,1].mean()/512),
                              repeated_identifier_bpb=float(v[:,3].sum()/v[:,2].sum()))
    write("result.json",dict(decision=decision,phase=phase,validation_interval=ci,models=stats,validation_control=pre,
                        test_control=test_control,steps=end,seed=order,seconds_this_invocation=time.monotonic()-started,
                        device=mx.device_info()["device_name"]))
if __name__=="__main__":main()

