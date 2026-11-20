"""Stress one early learned write with a long sequence of near-miss sensor events."""
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import mlx.core as mx
import numpy as np

from volume import neighbors

HERE=Path(__file__).resolve().parent
SEEDS=(101,211,307)
SIDE=16
DEPTH=128
TIMEPOINTS=(16,128)


def scenario(rng,batch,side,depth):
    shape=(batch,side,side,side)
    words=rng.choice(np.array([-1,1],np.float32),(*shape,32))
    actions=rng.integers(7,size=(depth-1,*shape),dtype=np.int32)
    blue=rng.integers(2,size=actions.shape,dtype=np.int8)
    local=1-np.roll(blue,1,axis=4)
    save=rng.integers(2,size=shape,dtype=np.int8)
    blue[7]=1
    local[7]=save
    partner=local.copy()
    partner[7]=1-save
    origin=np.broadcast_to(np.arange(side**3,dtype=np.int32).reshape(1,side,side,side),shape)
    transported=origin.copy()
    for command in actions[:7]:
        transported=np.choose(command,neighbors(transported,np))
    truth=[]
    for s in (save,1-save):
        index=np.where(s,transported,origin).reshape(batch,-1,1)
        value=np.take_along_axis(words.reshape(batch,-1,32),index,axis=1).reshape(words.shape)
        truth.append(value)
    changed=np.any(truth[0]!=truth[1],axis=-1)
    return words,actions,blue,local,partner,truth,changed


def rollout(weights,words,actions,blue,local,partner,hard,timepoints):
    read=mx.softmax(mx.array(weights["read"],dtype=mx.float32),axis=1)
    affine=mx.array(weights["gate"],dtype=mx.float32)
    bias=mx.array(weights["bias"],dtype=mx.float32)
    work=memory=mx.concatenate((mx.array(words),mx.array(words)),axis=0)
    output={}
    for step,(a,b,x,y) in enumerate(zip(actions,blue,local,partner,strict=True),start=1):
        old=work
        commands=mx.concatenate((mx.array(a),mx.array(a)),axis=0)
        p=read[commands]
        work=sum(p[...,i:i+1]*v for i,v in enumerate(neighbors(old,mx)))
        b=mx.array(b,dtype=mx.float32)
        inputs=mx.stack(neighbors(b,mx),axis=-1)
        remote=inputs@affine[1:]+bias
        local_a=mx.array(x,dtype=mx.float32)
        local_b=mx.array(y,dtype=mx.float32)
        logit=mx.concatenate((local_a*affine[0]+remote,local_b*affine[0]+remote),axis=0)
        gate=(logit>0).astype(mx.float32) if hard else mx.sigmoid(logit)
        memory=memory+gate[...,None]*(old-memory)
        if step+1 in timepoints:
            restored=sum(read[7,i]*v for i,v in enumerate((*neighbors(work,mx),memory)))
            output[step+1]=np.array(restored)>0
        if step%8==0:
            mx.eval(work,memory)
    return output


def evaluate(write,repeats,batch,side,depth,timepoints):
    rng=np.random.default_rng(867382)
    data=[]
    for _ in range(repeats):
        w,a,b,x,y,truth,changed=scenario(rng,batch,side,depth)
        records=[changed.sum((1,2,3))]
        for hard in (False,True):
            values=rollout(write,w,a,b,x,y,hard,timepoints)
            gold=np.concatenate(truth,axis=0)>0
            for t in timepoints:
                right=(values[t]==gold).all(-1)
                u,v=np.split(right,2,axis=0)
                records.append((u&v&changed).sum(axis=(1,2,3)))
        data.append(np.stack(records,axis=1))
    return np.concatenate(data)


def summarize(data):
    totals=data.sum(1)
    names=("soft_short","soft_long","hard_short","hard_long")
    seed_summary={str(seed):dict(eligible=int(row[0]),correct=dict(zip(names,row[1:].tolist(),strict=True)))
                  for seed,row in zip(SEEDS,totals,strict=True)}
    combined=totals.sum(0)
    rate=combined[1:]/combined[0]
    draws=np.random.default_rng(341509).integers(data.shape[1],size=(4000,data.shape[1]))
    pair=data.sum(0)
    effect=(pair[:,4]-pair[:,2])
    interval=np.percentile(effect[draws].sum(1)/pair[:,0][draws].sum(1),[2.5,97.5]).tolist()
    passes=rate[0]>=.95 and rate[3]-rate[1]>=.05 and interval[0]>0
    return dict(decision="long_hold_leakage" if passes else "retention_measured",
                by_seed=seed_summary,pooled_rates=dict(zip(names,rate.tolist(),strict=True)),
                hard_minus_soft_long=dict(mean=float(rate[3]-rate[1]),interval=interval))


def main():
    start=time.perf_counter()
    out=Path(os.environ["STUDIO_JOB_DIR"])
    mx.set_cache_limit(2*1024**3)
    runs=[]
    weights={}
    for seed in SEEDS:
        path=HERE/"results"/f"cube-full-{seed}.json"
        weights[str(seed)]=sha256(path.read_bytes()).hexdigest()
        result=evaluate(json.loads(path.read_text()),16,4,SIDE,DEPTH,TIMEPOINTS)
        runs.append(result)
        print(f"Seed {seed}: {result.sum(0).tolist()}",flush=True)
    samples=np.array(runs)
    np.save(out/"per-volume.npy",samples)
    report=dict(side=SIDE,voxels=SIDE**3,bits=32,timepoints=TIMEPOINTS,volumes=samples.shape[1],
                runtime_s=time.perf_counter()-start,peak_mlx_bytes=mx.get_peak_memory(),
                frozen_checkpoints=weights,**summarize(samples))
    (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
