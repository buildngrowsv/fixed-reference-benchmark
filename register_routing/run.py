from pathlib import Path
from functools import partial
import os
import sys
import time
import json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"reference_lm"))
from train import Model,Block,mx,nn,optim,optimizer,snapshot,restore,write,OUT,RUNS,scores,uncertainty
from programs import batch

def routing(ids,wrong):
    events=ids.reshape(len(ids),64,4)
    op=events[:,:,0];key=events[:,:,1];ix=mx.arange(64)
    # In row q, scatter valid positions p<q to (p+1)%q for the density-matched control.
    q=ix[:,None];p=ix[None,:]
    lookup=(p-1+mx.maximum(q,1))%mx.maximum(q,1) if wrong else mx.broadcast_to(p,(64,64))
    src_key=key[:,lookup];src_op=op[:,lookup]
    eligible=(src_op<2)&(src_key==key[:,:,None])&(p<q)[None,:,:]
    t=mx.arange(256);is_query=(op[:,t//4]>=2)&(t%4==3)[None,:]
    use=eligible[:,(t//4)[:,None],(t//4)[None,:]]&(t%4==3)[None,None,:]
    causal=(t[:,None]>=t[None,:]);self_edge=(t[:,None]==t[None,:])
    return mx.where(is_query[:,:,None],use|self_edge[None,:,:],causal[None,:,:])[:,None,:,:]

class RoutedModel(Model):
    def __call__(self,ids,fixed=False,wrong=False,mask=None,exchange=None):
        x=(self.tokens(ids)+self.pos(mx.arange(ids.shape[1]))).astype(mx.bfloat16)
        if mask is None:mask=routing(ids,wrong)
        for j,b in enumerate(self.blocks):
            if j!=3:x=b(x,False);continue
            n,t,d=x.shape
            q,k,v=[z.reshape(n,t,6,64).transpose(0,2,1,3) for z in mx.split(b.qkv(b.norm1(x).astype(x.dtype)),3,axis=-1)]
            a=mx.fast.scaled_dot_product_attention(q[:,:1],k[:,:1],v[:,:1],scale=.125,mask=mask)
            if exchange is not None:
                slots,partners=exchange
                target=(mx.arange(t)[None,:]==slots[:,None])[:,None,:,None]
                values=mx.where(target,v[partners,:1],v[:,:1])
                replaced=mx.fast.scaled_dot_product_attention(q[:,:1],k[:,:1],values,scale=.125,mask=mask)
                a=mx.concatenate([a[:,:,:-1],replaced[:,:,-1:]],axis=2)
            c=mx.fast.scaled_dot_product_attention(q[:,1:],k[:,1:],v[:,1:],scale=.125,mask="causal")
            x=x+b.out(mx.concatenate([a,c],axis=1).transpose(0,2,1,3).reshape(n,t,d))
            x=x+b.down(nn.gelu(b.up(b.norm2(x).astype(x.dtype))))
        return self.read(self.norm(x[:,3::4,:]).astype(x.dtype)).astype(mx.float32)

def evaluate(m,arm,seed):
    out=[]
    for j in range(32):
        ids,y,op,changed,guess=batch([seed,j])
        logits=m(mx.array(ids),False) if arm=="dense" else m(mx.array(ids),wrong=arm=="rotated")
        hit=np.array(logits.argmax(-1))==y
        row=[]
        for kind in (2,3):
            mask=changed&(op==kind)
            row.extend([mask.sum(1),(hit&mask).sum(1),((guess==y)&mask).sum(1)])
        out.append(np.stack(row,axis=1))
    return np.concatenate(out)

def main():
    seed=int(sys.argv[1])
    raw=batch([seed,39127],4)[0]
    a,b=[np.array(routing(mx.array(raw),w)) for w in (False,True)]
    assert np.array_equal(a.sum(-1),b.sum(-1)) and not np.triu(a,1).any() and not np.triu(b,1).any()
    dense=Model();restore(dense,optimizer(),RUNS/"reference-transformer-pilot"/"writable.safetensors")
    models={"dense":dense}
    for arm in ("matched","rotated"):
        mx.random.seed(seed);m=RoutedModel();opt=optimizer();wrong=arm=="rotated"
        path=OUT/(arm+".safetensors");first=restore(m,opt,path) if path.exists() else 0
        def loss(m,x,y,op):
            e=nn.losses.cross_entropy(m(x,wrong=wrong),y,reduction="none")
            mask=op>=2
            return (e*mask).sum()/mx.maximum(mask.sum(),1)
        vg=nn.value_and_grad(m,loss);state=[m.state,opt.state]
        @partial(mx.compile,inputs=state,outputs=state)
        def step(x,y,op):
            l,g=vg(m,x,y,op);g,_=optim.clip_grad_norm(g,1.);opt.update(m,g);return l
        since=time.monotonic()
        for t in range(first+1,513):
            ids,y,op,_,_=batch([seed,27491,t])
            l=step(mx.array(ids),mx.array(y),mx.array(op));mx.eval(l,state)
            if t==1 or t%64==0:
                snapshot(m,opt,t,path)
                write("progress.json",dict(arm=arm,step=t,target=512,time=time.time(),loss=l.item(),
                    tokens_per_second=(t-first)*8192/max(time.monotonic()-since,.001)))
        models[arm]=m
    metrics={};gate=False
    for label,s in (("validation",72718),("test",625137)):
        hits={k:evaluate(m,k,s) for k,m in models.items()}
        hits["matched_switched"]=evaluate(models["matched"],"rotated",s)
        np.savez_compressed(OUT/(label+".npz"),**hits)
        record={k:scores(v) for k,v in hits.items()}
        if label=="validation":
            gate=all(np.mean(record["matched"])-np.mean(record[k])>=.05 and uncertainty(hits["matched"],hits[k])[0]>0 for k in ("rotated","dense"))
        metrics[label]=record
    write("result.json",dict(decision="replicate" if gate and seed==101 else "complete",seed=seed,metrics=metrics))
if __name__=="__main__":main()

