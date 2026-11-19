"""Independent CPU replay of the held-out cube using flat node indices and saved weights."""
import json
import numpy as np
from pathlib import Path

R=Path(__file__).resolve().parent/"results"


def main():
    side=6
    coords=np.array(list(np.ndindex((side,side,side))))
    offsets=np.array([[0,0,0],[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]])
    adjacent=np.ravel_multi_index(((coords[:,None,:]+offsets)%side).transpose(2,0,1),(side,)*3)
    for seed in (101,211,307):
        fit=json.loads((R/f"cube-full-{seed}.json").read_text())
        gate=np.array(fit["gate"],dtype=np.float32)
        scores=np.array(fit["read"],dtype=np.float32)
        read=np.exp(scores-scores.max(1,keepdims=True));read/=read.sum(1,keepdims=True)
        rng=np.random.default_rng(105492)
        results=[]
        for _ in range(16):
            shape=(8,side,side,side)
            x=rng.choice(np.array([-1,1],np.float32),(*shape,32)).reshape(8,-1,32)
            a=rng.integers(8,size=(16,*shape),dtype=np.int32).reshape(16,8,-1)
            v=rng.integers(2,size=(16,*shape,3),dtype=np.int8).reshape(16,8,-1,3)
            h=x.copy();m=x.copy()
            start=np.repeat(np.arange(len(coords))[None,:],8,axis=0)
            src=start.copy();mem=start.copy();never=start.copy()
            for step,b in zip(a,v):
                old=h
                ab=np.concatenate([b[:,:,[0]],b[:,:,1][:,adjacent]],axis=2).astype(np.float32)
                g=1/(1+np.exp(-(ab@gate+fit["bias"])))
                w=read[step]
                h=sum(w[:,:,i,None]*old[:,adjacent[:,i]] for i in range(7))+w[:,:,7,None]*m
                m=m+g[:,:,None]*(old-m)
                choice=adjacent[np.arange(len(coords))[None,:],np.minimum(step,6)]
                prev=src
                src=np.where(step==7,mem,np.take_along_axis(src,choice,axis=1))
                never=np.where(step==7,start,np.take_along_axis(never,choice,axis=1))
                mem=np.where((ab[:,:,0]==1)&(ab[:,:,6]==1),prev,mem)
            y=np.take_along_axis(x,src[:,:,None],axis=1)
            mask=src!=never
            bit=(h>0)==(y>0)
            results.append(np.stack([mask.sum(1),(bit.all(-1)&mask).sum(1),(bit*mask[:,:,None]).sum((1,2))],1))
        with np.load(R/f"cube-{seed}.npz") as records:
            np.testing.assert_array_equal(np.concatenate(results),records["full_full"])
    print("Independent flat-index NumPy simulation agrees on all 128 volumes for each seed.")


if __name__=="__main__": main()
