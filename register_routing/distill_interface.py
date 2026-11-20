"""Distil the transferred 64-value signal and plug it into the frozen recipient."""
from functools import partial
import sys,time
import numpy as np
from prefix_surgery import source_head
from head_surgery import DONOR,load_model
from run import (OUT,RUNS,RoutedModel,Block,batch,mx,nn,optim,optimizer,restore,snapshot,
                 routing,evaluate,scores,uncertainty,write)
from train import Projection
from mlx.utils import tree_flatten

class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens=nn.Embedding(54,128);self.pos=nn.Embedding(256,128)
        self.context=Block(False,128);self.norm=nn.RMSNorm(128)
        self.read=Projection(128,3*64,bias=False)
    def __call__(self,x,exchange=None):
        h=(self.tokens(x)+self.pos(mx.arange(x.shape[1]))).astype(mx.bfloat16)
        h=self.context(h,False)
        q,k,v=mx.split(self.read(self.norm(h).astype(h.dtype)),3,axis=-1)
        mask=routing(x,True)
        attention=mx.fast.scaled_dot_product_attention(q[:,None],k[:,None],v[:,None],scale=.125,mask=mask)
        if exchange is None:return attention
        slots,partners=exchange
        site=(mx.arange(x.shape[1])[None,:]==slots[:,None])[:,:,None]
        changed=mx.where(site,v[partners],v)
        alt=mx.fast.scaled_dot_product_attention(q[:,None],k[:,None],changed[:,None],scale=.125,mask=mask)
        return mx.concatenate([attention[:,:,:-1],alt[:,:,-1:]],axis=2)

def objective(m,x,target,queries):
    error=mx.square(m(x).astype(mx.float32)-target.astype(mx.float32)).mean(axis=(1,3))
    return .5*(error.mean()+(error*queries).sum()/mx.maximum(queries.sum(),1))

def fit(donor,arm):
    mx.random.seed(2741);m=Student();opt=optimizer()
    dest=OUT/(arm+".safetensors")
    start=restore(m,opt,dest) if dest.exists() else 0
    vg=nn.value_and_grad(m,objective);state=[m.state,opt.state]
    @partial(mx.compile,inputs=state,outputs=state)
    def step(x,reference,use):
        loss,grad=vg(m,x,reference,use);grad,_=optim.clip_grad_norm(grad,1.)
        opt.update(m,grad);return loss
    then=time.monotonic()
    for t in range(start+1,4097):
        ids,_,ops,_,_=batch([73219,t]);x=mx.array(ids)
        ref=source_head(donor,donor,x)
        if arm=="permuted":ref=mx.roll(ref,1,axis=0)
        use=mx.array(np.repeat(ops>=2,4,axis=1)&(np.arange(256)%4==3))
        loss=step(x,ref,use);mx.eval(loss,state)
        if t%64==0:
            if not np.isfinite(loss.item()):raise ValueError("nonfinite loss")
            snapshot(m,opt,t,dest)
            write("progress.json",dict(arm=arm,step=t,target=4096,time=time.time(),loss=loss.item(),
                  tokens_per_second=(t-start)*8192/(time.monotonic()-then)))
    return m

def main():
    smoke=sys.argv[1:]==["smoke"]
    if smoke:mx.set_default_device(mx.cpu)
    donor=load_model(DONOR)
    if smoke:
        ids,_,ops,_,_=batch([930,0],2);x=mx.array(ids)
        s=Student();ref=source_head(donor,donor,x)
        q=mx.array(np.repeat(ops>=2,4,axis=1)&(np.arange(256)%4==3))
        val,g=nn.value_and_grad(s,objective)(s,x,ref,q)
        assert ref.shape==s(x).shape==(2,1,256,64) and np.isfinite(val.item())
        assert all(np.isfinite(np.array(v)).all() for _,v in tree_flatten(g))
        print("Teacher, student, loss and gradients agree on interface shape")
        return
    host=RoutedModel();assert restore(host,optimizer(),RUNS/"displaced-prefix-surgery/trained_prefix.safetensors")==1024
    signals={"teacher":lambda x:source_head(donor,donor,x)}
    for arm in ("aligned","permuted"):signals[arm]=fit(donor,arm)
    def bind(fn):
        return lambda x,wrong=True:host(x,wrong=wrong,first_head=fn(x))
    models={k:bind(v) for k,v in signals.items()}
    counts={k:evaluate(m,"rotated",319754) for k,m in models.items()}
    np.savez_compressed(OUT/"test.npz",**counts)
    accuracy={k:scores(v) for k,v in counts.items()};effects={}
    for k in ("teacher","permuted"):
        effects[k]=dict(delta=float(np.mean(accuracy["aligned"])-np.mean(accuracy[k])),
                        interval=uncertainty(counts["aligned"],counts[k]))
    weights=sum(v.size for _,v in tree_flatten(signals["aligned"].parameters()))
    full=(54+256)*384+3*(12*384*384+2*384)+384+3*384*64
    x=mx.array(batch([319754,1001])[0]);latency={k:[] for k in ("teacher","aligned")}
    timed={k:mx.compile(models[k]) for k in latency}
    for fn in timed.values():mx.eval(fn(x));mx.eval(fn(x))
    for i in range(20):
        for k in (("teacher","aligned") if i%2==0 else ("aligned","teacher")):
            then=time.perf_counter();mx.eval(timed[k](x));latency[k].append(time.perf_counter()-then)
    good=effects["teacher"]["interval"][0]>-.05 and effects["permuted"]["delta"]>=.10 and effects["permuted"]["interval"][0]>0 and weights<=.10*full
    write("result.json",dict(decision="replicate" if good else "stop",accuracy=accuracy,effects=effects,
          student_weights=weights,teacher_weights_used=full,batch_seconds=latency))

if __name__=="__main__":main()
