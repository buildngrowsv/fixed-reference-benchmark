"""Use a compiled detector sampler in the recurrent simulation and verify parity."""
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import mlx.core as mx
import numpy as np

from anchored_witness_routing import ANCHORS
from device_witness_timing import line_indices
from inferred_regions import PROBES, ROUNDS, evaluate
from long_hold import HERE, SEEDS
from probe_scale import SIDE, WORLD_IDS
from spatial_witness import recorded_step

PRIOR = "spatial-anchored-witness-routing-7077888"
REVISION = "813daccd502125d79703bff8568755a65bdbade5"


def observer(side, probes, anchors, distance):
    half=side//2
    lines=line_indices(side,probes)
    origin=np.array(list(np.ndindex(2,2,2)))[:,None,:]*half
    candidates=[anchors]
    for axis in range(3):
        other=anchors.copy()
        other[...,axis]=(other[...,axis]-distance)%half
        candidates.append(other)
    points=mx.array((np.stack(candidates)+origin).astype(np.int32))
    stride=mx.array([side**2,side,1],dtype=mx.int32)
    mx.eval(lines,points,stride)

    @mx.compile
    def observe(field):
        flat=field.reshape(-1)
        edge=mx.take(flat,lines)
        disagreements=mx.sum(edge!=mx.roll(edge,1,axis=1),axis=2)
        shift=mx.argmax(disagreements[:,:half]+disagreements[:,half:],axis=1)
        coords=(points+shift)%side
        selected=mx.take(flat,mx.sum(coords*stride,axis=-1))
        matches=mx.sum(selected[0]&selected[1:],axis=-1)
        return mx.concatenate((shift,mx.argmin(matches,axis=0)))

    def read(field):
        return np.array(observe(field))

    return read


def main():
    start=time.perf_counter()
    mx.set_cache_limit(2*1024**3)
    out=Path(os.environ["STUDIO_JOB_DIR"])
    root=Path(os.environ["STUDIO_QUEUE_ROOT"])/"runs"/PRIOR
    receipt=json.loads((root/"receipt.json").read_text())
    assert (receipt["revision"],receipt["status"],receipt["exit"])==(REVISION,"done",0)
    prior=json.loads((root/"result.json").read_text())
    assert prior["decision"]=="dedicated_probes_scale"
    seed=SEEDS[0]
    weights=(HERE/"results"/f"cube-full-{seed}.json").read_bytes()
    signature=sha256(weights).hexdigest()
    assert signature==prior["frozen_checkpoints"][str(seed)]
    step=recorded_step(json.loads(weights))
    all_counts,all_shifts,all_decisions=[],[],[]
    for world in WORLD_IDS:
        stable=np.random.default_rng(918447+world).integers(SIDE//2,size=(8,ANCHORS,3))
        logged=[]
        def route(probes):
            read=observer(SIDE,probes,stable,8)
            def capture(field):
                answer=read(field)
                logged.append(answer)
                return answer
            return capture
        counts,shifts=evaluate(step,world,side=SIDE,stratified=True,device_router=route)
        assert len(logged)==ROUNDS*4
        all_counts.append(counts)
        all_shifts.append(shifts)
        all_decisions.append(np.stack(logged))
        print(f"Completed world {world} for checkpoint {seed}",flush=True)
    counts,shifts,decisions=map(np.stack,(all_counts,all_shifts,all_decisions))
    original=root/"worlds.npz"
    with np.load(original) as earlier:
        match=bool(np.array_equal(counts,earlier["counts"][0]) and np.array_equal(shifts,earlier["shifts"][0]))
    np.savez_compressed(out/"parity.npz",counts=counts,shifts=shifts,decisions=decisions)
    result=dict(decision="compiled_observer_exact_parity" if match else "unresolved",
                cells=SIDE**3,checkpoint=seed,checkpoint_digest=signature,worlds=list(WORLD_IDS),
                source_archive_digest=sha256(original.read_bytes()).hexdigest(),transitions=ROUNDS*16,
                sampled_scalars=3*SIDE*PROBES+4*8*ANCHORS,observer_output_scalars=11,
                decisions=int(np.prod(decisions.shape[:-1])),counts_and_shifts_exact=match,
                elapsed_s=time.perf_counter()-start,peak_mlx_bytes=mx.get_peak_memory())
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
