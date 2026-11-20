"""Test first-round adaptation to independently timed and oriented boundary shifts."""
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import mlx.core as mx
import numpy as np

from anchored_witness_routing import ANCHORS,routing
from inferred_regions import PROBES,ROUNDS,evaluate,summarize
from long_hold import HERE,SEEDS
from probe_scale import SIDE
from spatial_witness import recorded_step

WORLDS=range(48,56)


def schedule(world):
    rng=np.random.default_rng(475882+world)
    times=np.arange(4,13,4)+rng.integers(-1,2,size=3)
    axes=rng.permutation(3)
    sizes=rng.choice([28,36,44,52,60,68],size=3,replace=False)
    return {int(t):int(s)*np.eye(3,dtype=np.int64)[a] for t,s,a in zip(times,sizes,axes,strict=True)}


def immediate(counts, plans):
    worlds=np.stack([counts[i,list(p),:,1] for i,p in enumerate(plans)])
    totals=worlds.sum(axis=(0,1))
    point=totals[:,3]/totals[:,2]
    legitimate=totals[:,4]/totals[:,0]
    draws=np.random.default_rng(976131).integers(len(plans),size=(5000,len(plans)))
    bags=worlds[draws].sum(axis=(1,2))
    ratios=bags[...,3]/bags[...,2]
    changes={}
    for label,arm in (("sampled_minus_stale",0),("sampled_minus_known",2)):
        delta=ratios[:,1]-ratios[:,arm]
        changes[label]=dict(mean=float(point[1]-point[arm]),interval=np.percentile(delta,[2.5,97.5]).tolist())
    stale,known=changes["sampled_minus_stale"],changes["sampled_minus_known"]
    qualified=bool(stale["mean"]>=.04 and stale["interval"][0]>0 and known["interval"][0]>=-.02
                   and legitimate[1]>=.98)
    return dict(retention=point.tolist(),legitimate=legitimate.tolist(),changes=changes,qualified=qualified)


def main():
    begin=time.perf_counter()
    mx.set_cache_limit(2*1024**3)
    out=Path(os.environ["STUDIO_JOB_DIR"])
    seed=SEEDS[0]
    payload=(HERE/"results"/f"cube-full-{seed}.json").read_bytes()
    model=recorded_step(json.loads(payload))
    plans=[schedule(w) for w in WORLDS]
    results=[]
    for w,p in zip(WORLDS,plans,strict=True):
        results.append(evaluate(model,w,side=SIDE,stratified=True,sampled_routes=routing(w,SIDE),boundary_steps=p))
        print(f"Completed staggered world {w}",flush=True)
    counts,edges=(np.stack(x) for x in zip(*results,strict=True))
    for c,s,plan in zip(counts,edges,plans,strict=True):
        actual=s[:,1,1]
        current=actual[0]
        for i,value in enumerate(actual):
            if i in plan:
                current=(current+plan[i])%(SIDE//2)
            np.testing.assert_array_equal(value,current)
        initial=min(plan)
        np.testing.assert_array_equal(c[:initial,0],c[:initial,2])
    fast=immediate(counts,plans)
    all_late=summarize(counts[None],edges[None],SIDE)
    np.savez_compressed(out/"staggered.npz",counts=counts,shifts=edges)
    result=dict(decision="sparse_router_recovers_staggered_jumps" if fast["qualified"] else "unresolved",
                cells=SIDE**3,worlds=list(WORLDS),checkpoint=seed,
                checkpoint_digest=sha256(payload).hexdigest(),
                schedules=[{t:v.tolist() for t,v in p.items()} for p in plans],
                first_round=fast,latter_half=all_late,sampled_fraction=3*PROBES/SIDE**2+4*8*ANCHORS/SIDE**3,
                runtime_s=time.perf_counter()-begin,peak_mlx_bytes=mx.get_peak_memory())
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
