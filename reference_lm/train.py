"""Matched bottom-up transformer fits with durable optimizer and model checkpoints."""
from functools import partial
from pathlib import Path
import json
import os
import sys
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten,tree_unflatten
import numpy as np
from programs import batch

OUT=Path(os.environ["STUDIO_JOB_DIR"])
RUNS=Path(os.environ["STUDIO_QUEUE_ROOT"])/"runs"
WIDTH=384
ARMS=("fixed","writable")

def write(name,d):
    p=OUT/name;tmp=p.with_suffix(".tmp");tmp.write_text(json.dumps(d,indent=2)+"\n");os.replace(tmp,p)

class Projection(nn.Linear):
    def __call__(self,x):
        return x@self.weight.astype(x.dtype).T

class Block(nn.Module):
    def __init__(self,protected,width=WIDTH):
        super().__init__()
        self.qkv=Projection(width,3*width,bias=False);self.out=Projection(width,width,bias=False)
        self.up=Projection(width,4*width,bias=False);self.down=Projection(4*width,width,bias=False)
        self.norm1=nn.RMSNorm(width);self.norm2=nn.RMSNorm(width)
        if protected:
            for proj in (self.out,self.down):
                proj.weight=mx.concatenate([proj.weight[:-64],mx.zeros_like(proj.weight[-64:])])
    def __call__(self,x,protect):
        n,t,d=x.shape
        q,k,v=mx.split(self.qkv(self.norm1(x).astype(x.dtype)),3,axis=-1)
        def heads(z):return z.reshape(n,t,d//64,64).transpose(0,2,1,3)
        attended=mx.fast.scaled_dot_product_attention(heads(q),heads(k),heads(v),scale=.125,mask="causal")
        def add(h,inc):
            if protect:return mx.concatenate([h[...,:-64]+inc[...,:-64],h[...,-64:]],axis=-1)
            return h+inc
        x=add(x,self.out(attended.transpose(0,2,1,3).reshape(n,t,d)))
        return add(x,self.down(nn.gelu(self.up(self.norm2(x).astype(x.dtype)))))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens=nn.Embedding(54,WIDTH);self.pos=nn.Embedding(256,WIDTH)
        self.blocks=[Block(2<=j<7) for j in range(8)]
        self.read=Projection(WIDTH,32,bias=False);self.norm=nn.RMSNorm(WIDTH)
    def __call__(self,ids,fixed,swap=False):
        x=(self.tokens(ids)+self.pos(mx.arange(ids.shape[1]))).astype(mx.bfloat16)
        for j,b in enumerate(self.blocks):
            if j==7 and swap:
                a,c=mx.split(x[...,-64:],2,axis=0)
                x=mx.concatenate([x[...,:-64],mx.concatenate([c,a],axis=0)],axis=-1)
            x=b(x,fixed and 2<=j<7)
        return self.read(self.norm(x[:,3::4,:]).astype(x.dtype)).astype(mx.float32)

def optimizer():
    schedule=lambda s:.00045*mx.minimum((s.astype(mx.float32)+1)/128,1.)
    return optim.AdamW(learning_rate=schedule,weight_decay=.03,bias_correction=True)

def snapshot(model,opt,step,file):
    weights={"weights."+k:v for k,v in tree_flatten(model.parameters())}
    weights.update({"optimizer."+k:v for k,v in tree_flatten(opt.state)})
    weights["step"]=mx.array(step)
    tmp=file.with_name(file.stem+".next.safetensors")
    mx.save_safetensors(str(tmp),weights);os.replace(tmp,file)

def restore(model,opt,file):
    data=mx.load(str(file))
    model.load_weights([(k[8:],v) for k,v in data.items() if k.startswith("weights.")])
    opt.state=tree_unflatten([(k[10:],v) for k,v in data.items() if k.startswith("optimizer.")])
    return int(data["step"].item())

def evaluate(m,arm,panel_seed,nbatches):
    counts=[]
    for i in range(nbatches):
        ids,y,op,changed,guess=batch([panel_seed,i])
        pred=np.array(m(mx.array(ids),arm=="fixed").argmax(-1))
        current=[]
        for cat in (2,3):
            mask=changed&(op==cat)
            current.extend([mask.sum(1),((pred==y)&mask).sum(1),((guess==y)&mask).sum(1)])
        counts.append(np.stack(current,axis=1))
    return np.concatenate(counts)
def scores(v):
    a=v.sum(0);return [float(a[1]/a[0]),float(a[4]/a[3])]
def intervene(m,arm):
    trials=[]
    for i in range(64):
        a,b=batch([958129,i],16,True)
        ids=mx.array(np.concatenate((a[0],b[0])))
        clean=np.array(m(ids,arm=="fixed").argmax(-1))[:,-1]
        swap=np.array(m(ids,arm=="fixed",True).argmax(-1))[:,-1]
        gold=np.concatenate([a[1][:,-1],b[1][:,-1]])
        trials.append(np.stack([np.tile(a[2][:,-1],2),gold,clean,swap],axis=1))
    return np.concatenate(trials)

def uncertainty(a,b):
    rng=np.random.default_rng(192841)
    diffs=[]
    for _ in range(40):
        idx=rng.integers(len(a),size=(100,len(a)))
        x,y=a[idx].sum(1),b[idx].sum(1)
        diffs.extend(.5*(x[:,1]/x[:,0]-y[:,1]/y[:,0]+x[:,4]/x[:,3]-y[:,4]/y[:,3]))
    return np.quantile(diffs,[.025,.975]).tolist()

def main():
    phase,seed,parent=sys.argv[1],int(sys.argv[2]),sys.argv[3]
    end=512 if phase=="pilot" else 4096
    valid={};test={};meta={};beg=time.monotonic()
    for arm in ARMS:
        mx.random.seed(seed)
        model=Model();opt=optimizer()
        dest=OUT/(arm+".safetensors")
        prev=RUNS/parent/(arm+".safetensors")
        start=restore(model,opt,dest) if dest.exists() else (restore(model,opt,prev) if parent!="none" else 0)
        fixed=arm=="fixed"
        def loss(m,ids,target,op):
            err=nn.losses.cross_entropy(m(ids,fixed),target,reduction="none")
            mask=op>=2
            return (err*mask).sum()/mx.maximum(mask.sum(),1)
        grad=nn.value_and_grad(model,loss)
        state=[model.state,opt.state]
        @partial(mx.compile,inputs=state,outputs=state)
        def step(ids,labels,op):
            value,grads=grad(model,ids,labels,op)
            grads,_=optim.clip_grad_norm(grads,1.0)
            opt.update(model,grads)
            return value
        since=time.monotonic()
        for t in range(start+1,end+1):
            ids,y,op,_,_=batch([seed,27491,t])
            value=step(mx.array(ids),mx.array(y),mx.array(op))
            mx.eval(value,state)
            if t==1 or t%32==0:
                snapshot(model,opt,t,dest)
                speed=(t-start)*32*256/max(time.monotonic()-since,.001)
                write("progress.json",dict(arm=arm,step=t,target=end,seed=seed,loss=value.item(),tokens_per_second=speed,
                    time=time.time(),parameters=sum(v.size for _,v in tree_flatten(model.parameters()))))
                print(arm,t,round(value.item(),4),round(speed),"tokens/s",flush=True)
        valid[arm]=evaluate(model,arm,882191,32)
        np.save(OUT/(arm+"-validation.npy"),valid[arm])
        meta[arm]=dict(validation=scores(valid[arm]),parameters=sum(v.size for _,v in tree_flatten(model.parameters())))
        if phase!="pilot":
            test[arm]=evaluate(model,arm,628159,64)
            np.save(OUT/(arm+"-test.npy"),test[arm])
            np.save(OUT/(arm+"-intervention.npy"),intervene(model,arm))
            meta[arm]["test"]=scores(test[arm])
    diff=float(np.mean(scores(valid["fixed"]))-np.mean(scores(valid["writable"])))
    ci=uncertainty(valid["fixed"],valid["writable"])
    if phase=="pilot":decision="scale" if max(np.mean(scores(v)) for v in valid.values())>=.13125 else "floor"
    elif phase=="main" and abs(diff)>=.02 and ci[0]*ci[1]>0:decision="replicate"
    else:decision="complete"
    write("result.json",dict(decision=decision,seed=seed,phase=phase,models=meta,validation_fixed_minus_writable=diff,paired_interval=ci,
                            seconds_this_invocation=time.monotonic()-beg,device=mx.device_info()["device_name"]))
if __name__=="__main__":main()

