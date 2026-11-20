import sys
from pathlib import Path
import numpy as np
from functools import partial
from run import (RoutedModel,OUT,RUNS,batch,mx,nn,optim,optimizer,restore,snapshot,scores,uncertainty,write)
from run import routing

def random_mask(ids,seed):
    events=ids.reshape(len(ids),64,4)
    op,key=events[:,:,0],events[:,:,1]
    ix=np.arange(64)
    before=ix[None,:]<ix[:,None]
    eligible=(op[:,None,:]<2)&(key[:,:,None]==key[:,None,:])&before
    count=eligible.sum(axis=-1)
    draws=np.random.default_rng(seed).random(eligible.shape)
    ranks=np.argsort(np.argsort(np.where(before,draws,2),axis=-1),axis=-1)
    chosen=(ranks<count[:,:,None])&before
    assert np.array_equal(chosen.sum(2),count)
    return mx.array(chosen)

def token_mask(ids,chosen):
    t=mx.arange(256);ev=t//4
    allowed=chosen[:,ev[:,None],ev[None,:]]&(t%4==3)[None,None,:]
    isquery=(ids[:,4*ev]>=2)&(t%4==3)[None,:]
    return mx.where(isquery[:,:,None],allowed|(t[:,None]==t[None,:]),(t[:,None]>=t[None,:]))[:,None]

def main():
    mx.random.seed(101);m=RoutedModel();opt=optimizer()
    path=OUT/"random.safetensors"
    start=restore(m,opt,path) if path.exists() else 0
    def loss(m,ids,ch,y,op):
        err=nn.losses.cross_entropy(m(ids,mask=token_mask(ids,ch)),y,reduction="none")
        selected=op>=2
        return (err*selected).sum()/mx.maximum(selected.sum(),1)
    vg=nn.value_and_grad(m,loss);state=[m.state,opt.state]
    @partial(mx.compile,inputs=state,outputs=state)
    def step(x,ch,y,op):
        l,g=vg(m,x,ch,y,op);g,_=optim.clip_grad_norm(g,1.);opt.update(m,g);return l
    for t in range(start+1,513):
        ids,y,op,_,_=batch([101,27491,t]);x=mx.array(ids);ch=random_mask(ids,[2983,t])
        if t==1:
            a,b=np.array(token_mask(x,ch)),np.array(routing(x,False))
            assert np.array_equal(a.sum(-1),b.sum(-1)) and not np.triu(a,1).any()
        l=step(x,ch,mx.array(y),mx.array(op));mx.eval(l,state)
        if t%64==0:
            if not np.isfinite(l.item()):raise ValueError("nonfinite loss")
            snapshot(m,opt,t,path)
    matched=RoutedModel();restore(matched,optimizer(),RUNS/"register-addressing"/"matched.safetensors")
    results={}
    for phase,seed in (("validation",72718),("test",625137)):
        rows={"random":[],"matched_random":[]}
        for j in range(32):
            ids,y,op,changed,guess=batch([seed,j]);x=mx.array(ids)
            mask=token_mask(x,random_mask(ids,[59131,seed,j]))
            for name,model in (("random",m),("matched_random",matched)):
                hit=np.array(model(x,mask=mask).argmax(-1))==y
                cols=[]
                for kind in (2,3):
                    use=changed&(op==kind)
                    cols.extend([use.sum(1),(hit&use).sum(1),((guess==y)&use).sum(1)])
                rows[name].append(np.stack(cols,axis=1))
        cols={k:np.concatenate(v) for k,v in rows.items()}
        with np.load(RUNS/"register-addressing"/(phase+".npz")) as arch:
            cols.update({k:arch[k] for k in ("dense","rotated","matched")})
        for v in cols.values():np.testing.assert_array_equal(v[:,[0,2,3,5]],cols["random"][:,[0,2,3,5]])
        np.savez_compressed(OUT/(phase+".npz"),**cols)
        vals={k:scores(v) for k,v in cols.items()}
        results[phase]=dict(arms=vals,stable_minus_random=float(np.mean(vals["rotated"])-np.mean(vals["random"])),
            interval=uncertainty(cols["rotated"],cols["random"]))
    v=results["validation"]
    write("result.json",dict(decision="complete",supported=bool(v["stable_minus_random"]>=.10 and v["interval"][0]>0),metrics=results))
if __name__=="__main__":main()
