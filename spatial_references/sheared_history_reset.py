"""Test whether two successive observations can reset stale tilted-boundary evidence."""
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

import mlx.core as mx
import numpy as np

from inferred_regions import boundary_profiles, evaluate, witness_field
from long_hold import HERE, SEEDS
from probe_scale import SIDE
from sheared_witness import TiltedWorld, infer_shear, paired, warp
from spatial_witness import recorded_step

WORLDS=range(80,88)
JUMP=8


def circular_distance(a,b,period):
    d=np.abs(a-b)
    return np.minimum(d,period-d)


def reset_axes(current, prior, memory, period):
    return ((circular_distance(current,memory,period)>=12)
            & (circular_distance(prior,memory,period)>=12)
            & (circular_distance(current,prior,period)<=4))


class MovingWorld(TiltedWorld):
    def __init__(self,world,side,slope):
        super().__init__(world,side,slope)
        self.persistent=[None,None]
        self.reset=[None,None]
        self.previous=[None,None]
        self.slope=[None,None]
        self.offsets=[]
        self.fired=[]

    def observe(self,faults,previous,known,probes,distance):
        votes=[[],[],[]]
        selected=[]
        fit=[]
        triggers=[]
        visible=len(self.fitted)%2==1
        for k,(scene,active) in enumerate(zip(faults,previous,strict=True)):
            m=infer_shear(active,probes[0])
            fit.append(m)
            framed=warp(active,-m)
            profile=boundary_profiles(framed,probes)
            current=profile.argmax(1)
            clear=np.zeros(3,dtype=bool)
            if visible:
                if self.slope[k]!=m:
                    self.persistent[k]=profile.copy()
                    self.reset[k]=profile.copy()
                    self.slope[k]=m
                else:
                    clear=reset_axes(current,self.previous[k].argmax(1),self.reset[k].argmax(1),profile.shape[1])
                    self.persistent[k]+=profile
                    self.reset[k]+=profile
                    self.reset[k][clear]=(self.previous[k]+profile)[clear]
                self.previous[k]=profile
            choices=(current,self.persistent[k].argmax(1),self.reset[k].argmax(1)) if visible else (current,)*3
            selected.append(choices)
            triggers.append(clear)
            for i,offset in enumerate(choices):
                axes=self.routing(framed,offset,probes,distance)
                canonical=witness_field(warp(scene,-m),None,offset,distance,selected=axes)
                votes[i].append(warp(canonical,m))
        if visible:
            self.offsets.append(selected)
            self.fired.append(triggers)
        self.fitted.append(fit)
        return np.array(votes),np.asarray(selected)[:,2,:]


def main():
    started=time.perf_counter()
    mx.set_cache_limit(2*1024**3)
    w=SEEDS[0]
    source=(HERE/"results"/f"cube-full-{w}.json").read_bytes()
    driver=recorded_step(json.loads(source))
    smoke=sys.argv[1:]==["--smoke"]
    if smoke:
        np.testing.assert_array_equal(reset_axes(np.array([48,3,91]),np.array([46,4,92]),np.array([0,2,2]),96),[True,False,False])
    side, rounds, distance, jump=(16,4,2,2) if smoke else (SIDE,16,8,JUMP)
    draws=np.random.default_rng(737614)
    slopes=draws.permutation([-1]*4+[1]*4)
    offsets=np.eye(3,dtype=np.int32)[draws.integers(3,size=8)]*draws.choice([28,36,44,52,60,68],8)[:,None]
    results=[]
    for world,m,move in zip(WORLDS,slopes,offsets,strict=True):
        policy=MovingWorld(world,side,int(m))
        shift=(move if not smoke else 4*(move!=0))
        counts, truth=evaluate(driver,world,side=side,rounds=rounds,distance=distance,stratified=True,
                               geometry=policy,boundary_steps={jump:shift})
        actual=truth[:,1]
        np.testing.assert_array_equal(actual[jump,1],(actual[jump-1,1]+shift)%(side//2))
        results.append((counts,actual,np.array(policy.offsets),np.array(policy.fired)))
        print(f"Completed moving tilt world {world}",flush=True)
        if smoke:
            assert counts.shape==(4,3,2,6) and np.array(policy.offsets).shape==(4,2,3,3)
            print("Two-hit detector and small-volume moving-wall replay verified")
            return
    counts,true,estimates,triggers=(np.stack(x) for x in zip(*results,strict=True))
    exact=(estimates==true[:,:,:,None,:]).all(-1)
    window=slice(JUMP+1,JUMP+4)
    held=counts[:,window].sum(1)
    before=counts[:,:JUMP].sum(1)
    rate=held[...,3]/held[...,2]
    prior=before[...,3]/before[...,2]
    better=paired(rate[:,2],rate[:,1])
    versus_frame=paired(rate[:,2],rate[:,0])
    guard=paired(prior[:,2],prior[:,1])
    post_exact=exact[:,window,1].mean(axis=(0,1))
    passes=(post_exact[2]>=.85 and post_exact[2]-post_exact[1]>=.2
            and better["mean"][1]>=.01 and better["interval"][1][0]>0
            and versus_frame["interval"][1][0]>=-.005
            and min(x[0] for x in guard["interval"])>=-.005 and int(triggers[:,:,0].sum())<=4)
    out=Path(os.environ["STUDIO_JOB_DIR"])
    np.savez_compressed(out/"reset.npz",counts=counts,true_offsets=true,estimated_offsets=estimates,resets=triggers)
    r=dict(decision="reset_recovers_moved_tilted_offsets" if passes else "unresolved",
           worlds=list(WORLDS),slopes=slopes.tolist(),changes=offsets.tolist(),jump_round_1based=JUMP+1,
           shifted_full_offset_accuracy=post_exact.tolist(),post_retention=rate.mean(0).tolist(),
           reset_minus_persistent=better,reset_minus_same_frame=versus_frame,pre_reset_minus_persistent=guard,
           aligned_resets=int(triggers[:,:,0].sum()),checkpoint_sha256=sha256(source).hexdigest(),
           seconds=time.perf_counter()-started,peak_mlx_bytes=mx.get_peak_memory())
    (out/"result.json").write_text(json.dumps(r,indent=2)+"\n")


if __name__=="__main__":
    main()
