"""Independent flat-index CPU replay of the first eight held-out cubes for every fit."""
import json
from pathlib import Path

import numpy as np

R=Path(__file__).resolve().parent/"results"


def softmax(x):
    e=np.exp(x-x.max(-1,keepdims=True))
    return e/e.sum(-1,keepdims=True)


def inputs():
    rng=np.random.default_rng(984731)
    n=8;side=6;vol=side**3
    x=rng.choice(np.array([-1,1],np.float32),(n,vol,32))
    a=rng.integers(8,size=(16,n,vol),dtype=np.int32)
    signals=rng.integers(2,size=(16,n,vol,3),dtype=np.int8)
    coordinates=np.array(list(np.ndindex((side,)*3)))
    offsets=np.array([[0,0,0],[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]])
    ix=np.ravel_multi_index(((coordinates[:,None,:]+offsets)%side).transpose(2,0,1),(side,)*3)
    origin=np.broadcast_to(np.arange(vol),(n,vol))
    h=m=unchanged=origin
    for t,(action,s) in enumerate(zip(a,signals)):
        if t==8:
            half_h,half_m=h,m
            relative=origin
            relative_m=origin+vol
        command=ix[np.arange(vol),np.minimum(action,6)]
        def advance(w,ref):
            return np.where(action==7,ref,np.take_along_axis(w,command,axis=1))
        fire=(s[:,:,0]!=0)&(s[:,ix[:,5],1]!=0)
        past=h
        h=advance(h,m)
        m=np.where(fire,past,m)
        unchanged=advance(unchanged,origin)
        if t>=8:
            prev=relative
            relative=advance(relative,relative_m)
            relative_m=np.where(fire,prev,relative_m)
    target=lambda src:np.take_along_axis(x,src[:,:,None],axis=1)
    distinct=half_h!=half_m
    uses=(relative>=vol)&np.take_along_axis(distinct,relative%vol,axis=1)
    return x,a,signals,ix,target(h),h!=unchanged,target(half_m),distinct,uses


def predict(w,data,mode,corrupt=None):
    x,a,v,ix=data[:4]
    n=len(x)
    h=m=x
    flip=np.random.default_rng(984731+773).permutation(8)<4
    for t,(act,s) in enumerate(zip(a,v)):
        if t==8:
            if mode=="exchange": h,m=np.where(flip[:,None,None],m,h),np.where(flip[:,None,None],h,m)
            half=(h,m)
            if corrupt==0: h=h[np.arange(n)^1]
            if corrupt==1: m=m[np.arange(n)^1]
        feat=np.concatenate([s[:,:,[0]],s[:,:,1][:,ix]],axis=2).astype(np.float32)
        old=h
        if mode=="assigned":
            p=softmax(w["read"])[act]
            h=sum(p[:,:,j,None]*old[:,ix[:,j]] for j in range(7))+p[:,:,7,None]*m
            g=1/(1+np.exp(-(feat@w["gate"]+w["bias"])))
            m=m+g[:,:,None]*(old-m)
        else:
            bank=np.concatenate((old[:,ix],m[:,ix]),axis=2)
            table=softmax(w["table"])[act]
            g=1/(1+np.exp(-(feat@w["gate"]+w["bias"])))
            mixture=table[:,:,:,0]+g[:,:,:,None]*(table[:,:,:,1]-table[:,:,:,0])
            out=np.einsum("nvdj,nvjb->nvdb",mixture,bank)
            h,m=out[:,:,0],out[:,:,1]
    output=h if mode=="assigned" else softmax(w["head"])[0]*h+softmax(w["head"])[1]*m
    return output,half


def count(z,y,mask):
    hit=(z>0)==(y>0)
    return np.stack([mask.sum(1),(hit.all(-1)&mask).sum(1),(hit*mask[:,:,None]).sum((1,2))],axis=1)


def main():
    data=inputs()
    _,_,_,_,target,primary,memory,distinct,uses=data
    discrepancies=[]
    for mode in ("assigned","neutral","exchange"):
        for seed in (101,211,307):
            rec=json.loads((R/f"{mode}-{seed}.json").read_text())
            w={k:np.array(val,dtype=np.float32) for k,val in rec["weights"].items()}
            z,half=predict(w,data,mode)
            result={"primary":count(z,target,primary),"from_memory":count(z,target,uses)}
            for k in range(2):
                result[f"probe{k}"]=count(half[k],memory,distinct)
                result[f"corrupt{k}"]=count(predict(w,data,mode,k)[0],target,uses)
            with np.load(R/f"{mode}-{seed}.npz") as observed:
                for key,values in result.items():
                    diff=np.abs(values-observed[key][:8])
                    np.testing.assert_array_equal(values[:,0],observed[key][:8,0])
                    if key=="primary": np.testing.assert_array_equal(values[:,1],observed[key][:8,1])
                    if diff.any(): discrepancies.append([mode,seed,key,diff.sum(0).tolist()])
            if mode!="assigned":
                renamed=dict(table=np.roll(w["table"][:,::-1],7,axis=-1),gate=w["gate"][:,::-1],
                             bias=w["bias"][::-1],head=w["head"][::-1])
                np.testing.assert_allclose(predict(renamed,data,mode)[0],z,rtol=2e-5,atol=2e-5)
    (R/"replay.json").write_text(json.dumps(dict(volumes_per_fit=8,conditions=9,
        count_discrepancies=discrepancies,global_lane_rename_check=True),indent=2)+"\n")
    print("CPU replay count discrepancies",discrepancies)
    print("Primary whole-word counts and all denominators reproduced exactly.")


if __name__=="__main__": main()
