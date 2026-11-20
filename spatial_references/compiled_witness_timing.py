"""Measure compiled observer latency against both equivalent detector estimators."""
import argparse
import json
import os
from pathlib import Path
import time

import mlx.core as mx
import numpy as np

from compiled_witness_parity import observer
from device_witness_timing import REPETITIONS, compare
from long_hold import SEEDS
from probe_scale import SIDE, WORLD_IDS


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    small=parser.parse_args().smoke
    mx.set_cache_limit(2*1024**3)
    started=time.perf_counter()
    side,worlds,reps=(16,(32,),3) if small else (SIDE,WORLD_IDS,REPETITIONS)
    elapsed,decisions,weights=compare(side,worlds,reps,compiled_factory=observer)
    assert (decisions == decisions[:,0:1]).all()
    med=np.median(elapsed,axis=1)
    ratios=(med[:,:2]/med[:,2:]).reshape(len(SEEDS),len(worlds),2,2)
    by_world=np.median(ratios,axis=(0,2))
    rng=np.random.default_rng(483417)
    draws=rng.integers(len(worlds),size=(5000,len(worlds)))
    bound=np.percentile(np.median(by_world[draws],axis=1),[2.5,97.5],axis=0)
    speed=np.median(by_world,axis=0)
    decision="compiled_observer_latency_improves" if (speed[1]>=1.25 and bound[0,1]>1) else "unresolved"
    report=dict(decision=decision,cells=side**3,frames=len(elapsed),repetitions=reps,worlds=list(worlds),
                identical_decisions=True,median_ms=(np.median(elapsed,axis=(0,1))/1e6).tolist(),
                speedup=dict(zip(("full_materialization","two_gathers"),speed.tolist(),strict=True)),
                paired_intervals=dict(zip(("full_materialization","two_gathers"),bound.T.tolist(),strict=True)),
                output_scalars=11,frozen_checkpoints=weights,elapsed_s=time.perf_counter()-started,
                peak_mlx_bytes=mx.get_peak_memory())
    if small:
        print(json.dumps(report))
        return
    out=Path(os.environ["STUDIO_JOB_DIR"])
    np.savez_compressed(out/"timing.npz",elapsed_ns=elapsed,decisions=decisions)
    (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
