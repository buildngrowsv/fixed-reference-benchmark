"""Pool offset profiles over time while controlling for the same inferred tilt."""
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

import mlx.core as mx
import numpy as np

from inferred_regions import boundary_profiles, witness_field
from long_hold import HERE, SEEDS
from probe_scale import SIDE
from sheared_witness import TiltedWorld, WORLDS, benchmark, infer_shear, paired, warp
from spatial_witness import recorded_step

PRIOR = "spatial-sheared-witness-7077888"
REVISION = "2315111632a69a9e901e01d9f4d796ce401045eb"


class PooledWorld(TiltedWorld):
    def __init__(self, world, side, slope):
        super().__init__(world, side, slope)
        self.history = [None, None]
        self.last_slope = [None, None]

    def observe(self, faults, previous, known, probes, distance):
        votes = [[], [], []]
        estimates, slopes = [], []
        valid_frame = len(self.fitted) % 2 == 1
        panels = zip(faults, previous, known, self.slopes, strict=True)
        for i,(current, active, true_offset, true_slope) in enumerate(panels):
            slope = infer_shear(active, probes[0])
            slopes.append(slope)
            framed = warp(active, -slope)
            profiles = boundary_profiles(framed, probes)
            instantaneous = profiles.argmax(1)
            pooled = instantaneous
            if valid_frame:
                if self.last_slope[i] != slope:
                    self.history[i] = np.zeros_like(profiles)
                    self.last_slope[i] = slope
                self.history[i] += profiles
                pooled = self.history[i].argmax(1)
            estimates.append(pooled)
            for arm,(m,offset) in enumerate(((slope,instantaneous),(slope,pooled),(true_slope,true_offset))):
                canonical = framed if m==slope else warp(active,-m)
                axis = self.routing(canonical,offset,probes,distance)
                signal = witness_field(warp(current,-m),None,offset,distance,selected=axis)
                votes[arm].append(warp(signal,m))
        self.fitted.append(slopes)
        return np.array(votes),np.stack(estimates)


def main():
    started=time.perf_counter()
    mx.set_cache_limit(2*1024**3)
    checkpoint=SEEDS[0]
    weights=(HERE/"results"/f"cube-full-{checkpoint}.json").read_bytes()
    driver=recorded_step(json.loads(weights))
    if sys.argv[1:]==["--smoke"]:
        base,_,fit=benchmark(driver,16,WORLDS[:1],[1],rounds=4,distance=2)
        new,_,observed=benchmark(driver,16,WORLDS[:1],[1],rounds=4,distance=2,geometry_type=PooledWorld)
        np.testing.assert_array_equal(new[:,:,0],base[:,:,1])
        np.testing.assert_array_equal(new[:,:,2],base[:,:,2])
        np.testing.assert_array_equal(fit,observed)
        print("Verified full small-volume replay with exact same-frame and supplied controls")
        return
    source=Path(os.environ["STUDIO_QUEUE_ROOT"])/"runs"/PRIOR
    receipt=json.loads((source/"receipt.json").read_text())
    assert (receipt["revision"],receipt["status"],receipt["exit"])==(REVISION,"done",0)
    original=json.loads((source/"result.json").read_text())
    assert original["decision"]=="unresolved" and original["shifted_shear_accuracy"]==1
    assert original["checkpoint_sha256"]==sha256(weights).hexdigest()
    assert original["worlds"]==list(WORLDS)
    path=source/"sheared.npz"
    slopes=original["true_shears"]
    counts,offsets,fit=benchmark(driver,SIDE,WORLDS,slopes,geometry_type=PooledWorld)
    with np.load(path) as older:
        np.testing.assert_array_equal(counts[:,:,0],older["counts"][:,:,1])
        np.testing.assert_array_equal(counts[:,:,2],older["counts"][:,:,2])
        np.testing.assert_array_equal(fit,older["fitted_shears"])
    late=counts[:,8:].sum(1)
    rates=late[...,3]/late[...,2]
    lift=paired(rates[:,1],rates[:,0])
    deficit=paired(rates[:,1],rates[:,2])
    coordinates=(offsets[:,8:,0]==offsets[:,8:,1])
    exact=float(coordinates[...,1,:].all(-1).mean())
    passes=(exact>=.90 and lift["mean"][1]>=.01 and lift["interval"][1][0]>0
            and deficit["interval"][1][0]>=-.0075 and lift["interval"][0][0]>=-.005)
    out=Path(os.environ["STUDIO_JOB_DIR"])
    np.savez_compressed(out/"pooled.npz",counts=counts,offsets=offsets,shears=fit)
    report=dict(decision="history_repairs_tilted_offsets" if passes else "unresolved",
                retention=rates.mean(0).tolist(),exact_tilted_offset_vectors=exact,
                improvement=lift,versus_supplied=deficit,worlds=list(WORLDS),true_shears=slopes,
                baseline_sha256=sha256(path.read_bytes()).hexdigest(),checkpoint_sha256=sha256(weights).hexdigest(),
                seconds=time.perf_counter()-started,peak_mlx_bytes=mx.get_peak_memory())
    (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
