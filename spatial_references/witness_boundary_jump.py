"""Test recovery after a known wall layout jumps halfway through the rollout."""
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import mlx.core as mx
import numpy as np

from anchored_witness_routing import ANCHORS, routing
from inferred_regions import PROBES, ROUNDS, evaluate, summarize
from long_hold import HERE, SEEDS
from probe_scale import SIDE, WORLD_IDS
from spatial_witness import recorded_step


def main():
    start=time.perf_counter()
    mx.set_cache_limit(2*1024**3)
    out=Path(os.environ["STUDIO_JOB_DIR"])
    seed=SEEDS[0]
    weights=(HERE/"results"/f"cube-full-{seed}.json").read_bytes()
    driver=recorded_step(json.loads(weights))
    panels=[]
    for world in WORLD_IDS:
        panels.append(evaluate(driver,world,side=SIDE,stratified=True,
                               sampled_routes=routing(world,SIDE),boundary_jump=True))
        print(f"Completed jump world {world}",flush=True)
    counts,edges=(np.stack(x)[None] for x in zip(*panels,strict=True))
    actual=edges[0,:,:,1]
    assert (actual[:,:,0]==0).all()
    np.testing.assert_array_equal(actual[:,8:,1],(actual[:,:8,1]+SIDE//4)%(SIDE//2))
    assert (counts[:,:,:8,0]==counts[:,:,:8,2]).all()
    summary=summarize(counts,edges,SIDE)
    new=counts[0,:,8:,1].sum(axis=1)
    oracle=counts[0,:,8:,2].sum(axis=1)
    draw=np.random.default_rng(818470).integers(len(WORLD_IDS),size=(4000,len(WORLD_IDS)))
    samples=[]
    for arm in (new,oracle):
        a=arm[draw].sum(axis=1)
        samples.append(a[...,3]/a[...,2])
    interval=np.percentile(samples[0]-samples[1],[2.5,97.5],axis=0)
    differences={case:dict(mean=summary["rates"]["inferred"][case]["retention"]-
                                summary["rates"]["known"][case]["retention"],
                           interval=interval[:,i].tolist())
                 for i,case in enumerate(("aligned","shifted"))}
    lift=summary["inferred_minus_fixed"]["shifted"]
    legit=min(r["legitimate"] for r in summary["rates"]["inferred"].values())
    passes=(interval[0,1]>=-.02 and lift["mean"]>=.04 and lift["interval"][0]>0 and legit>=.98)
    epochs=counts[0].reshape(len(WORLD_IDS),2,8,3,2,6).sum(axis=(0,2))
    np.savez_compressed(out/"jump.npz",counts=counts,shifts=edges)
    summary.update(decision="sparse_witness_recovers_boundary_jump" if passes else "unresolved",
                   mode="quarter_side_jump",cells=SIDE**3,checkpoint=seed,worlds=list(WORLD_IDS),
                   checkpoint_digest=sha256(weights).hexdigest(),sampled_minus_known=differences,
                   epochs=epochs.tolist(),sample_fraction_upper=3*PROBES/SIDE**2+4*8*ANCHORS/SIDE**3,
                   runtime_s=time.perf_counter()-start,peak_mlx_bytes=mx.get_peak_memory())
    (out/"result.json").write_text(json.dumps(summary,indent=2)+"\n")


if __name__=="__main__":
    main()
