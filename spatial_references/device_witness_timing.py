"""Compare full host materialization with direct reads of the dedicated detector samples."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import mlx.core as mx
import numpy as np

from anchored_witness_routing import ANCHORS, choose_axes
from inferred_regions import PROBES, distributed_positions, infer_with_probes
from long_hold import HERE, SEEDS
from probe_scale import SIDE, WORLD_IDS
from regional_witness import combine
from spatial_witness import fault_mask
from volume import neighbors

REPETITIONS = 12


def line_indices(side, probes):
    stride = np.array([side**2, side, 1])
    axes = ((0, 1, 2), (1, 0, 2), (2, 1, 0))
    lines = []
    for (axis, j, k), pixels in zip(axes, probes, strict=True):
        a, b = np.unravel_index(pixels, (side, side))
        index = np.arange(side)[:, None]*stride[axis] + a*stride[j] + b*stride[k]
        lines.append(index)
    return mx.array(np.stack(lines).astype(np.int32))


def anchors_indices(side, shift, anchors, distance):
    half = side // 2
    origins = np.array(list(np.ndindex(2, 2, 2)))[:, None, :] * half + shift
    coordinates = (origins + anchors) % side
    stride = np.array([side**2, side, 1])
    indices = [coordinates @ stride]
    for axis in range(3):
        partner = coordinates.copy()
        partner[..., axis] = (origins[..., axis] + (anchors[..., axis]-distance) % half) % side
        indices.append(partner @ stride)
    return mx.array(np.stack(indices).astype(np.int32))


def gathered(field, lines, anchors, distance):
    side = field.shape[0]
    half = side//2
    flattened = field.reshape(-1)
    samples = np.asarray(mx.take(flattened, lines))
    transitions = (samples != np.roll(samples, 1, axis=1)).mean(axis=2)
    shift = (transitions[:, :half] + transitions[:, half:]).argmax(axis=1)
    data = np.asarray(mx.take(flattened, anchors_indices(side, shift, anchors, distance)))
    axes = (data[0] & data[1:]).mean(axis=2).argmin(axis=0)
    return np.concatenate((shift, axes))


def materialized(field, probes, anchors, distance):
    active = np.array(field)
    shift = infer_with_probes(active, probes)
    return np.concatenate((shift, choose_axes(active, shift, anchors, distance)))


def detector_frames(world, side, weights):
    rng = np.random.default_rng(777123 + world)
    axes = rng.permutation([0, 0, 0, 1, 1, 1, 2, 2])
    synchronized = combine(np.stack([np.moveaxis(fault_mask(rng, side//2)[1], 2, a) for a in axes]))
    shift = rng.choice(np.arange(4, side//2, 4), size=3)
    pulse = np.stack((synchronized, np.roll(synchronized, shift, axis=(0,1,2))))
    raw_blue = rng.integers(2, size=pulse.shape, dtype=np.int8)
    blue = np.where(np.roll(pulse,-1,axis=-1),1,raw_blue)
    local = np.where(pulse,1,1-np.roll(blue,1,axis=-1))
    c = mx.array(weights["gate"],dtype=mx.float32)
    offsets = mx.stack(neighbors(mx.array(blue,dtype=mx.float32),mx),axis=-1)
    active = mx.array(local,dtype=mx.float32)*c[0]+offsets@c[1:]+float(weights["bias"])>0
    mx.eval(active)
    return active[0],active[1]


def compare(side, worlds, repetitions, *, compiled_factory=None):
    timings, results, checkpoints = [], [], {}
    for seed in SEEDS:
        payload = (HERE/"results"/f"cube-full-{seed}.json").read_bytes()
        checkpoints[str(seed)] = sha256(payload).hexdigest()
        for world in worlds:
            rng = np.random.default_rng(790133+world)
            probes = [distributed_positions(rng,side) for _ in range(3)]
            lines = line_indices(side,probes)
            mx.eval(lines)
            anchors = np.random.default_rng(918447+world).integers(side//2,size=(8,ANCHORS,3))
            compiled = compiled_factory(side,probes,anchors,8) if compiled_factory is not None else None
            for frame in detector_frames(world,side,json.loads(payload)):
                methods = (lambda: materialized(frame,probes,anchors,8),
                           lambda: gathered(frame,lines,anchors,8))
                if compiled is not None:
                    methods += (lambda: compiled(frame),)
                outputs = [fn() for fn in methods]
                for output in outputs[1:]:
                    np.testing.assert_array_equal(output,outputs[0])
                elapsed = np.zeros((repetitions,len(methods)),dtype=np.int64)
                for rep in range(repetitions):
                    for offset in range(len(methods)):
                        arm = (rep+offset)%len(methods)
                        start = time.perf_counter_ns()
                        observed = methods[arm]()
                        elapsed[rep,arm] = time.perf_counter_ns()-start
                        np.testing.assert_array_equal(observed,outputs[arm])
                timings.append(elapsed)
                results.append(np.stack(outputs))
            print(f"Timed world {world}, checkpoint {seed}",flush=True)
    return np.stack(timings),np.stack(results),checkpoints


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke",action="store_true")
    smoke = ap.parse_args().smoke
    mx.set_cache_limit(2*1024**3)
    side, worlds, reps = (16,(32,),2) if smoke else (SIDE,WORLD_IDS,REPETITIONS)
    ts, decisions, checkpoints = compare(side,worlds,reps)
    ratios = np.median(ts[...,0],axis=1) / np.median(ts[...,1],axis=1)
    by_world = np.median(ratios.reshape(len(SEEDS),len(worlds),2),axis=(0,2))
    draws = np.random.default_rng(591277).integers(len(worlds),size=(5000,len(worlds)))
    bound = np.percentile(np.median(by_world[draws],axis=1),[2.5,97.5])
    speed = float(np.median(by_world))
    output = dict(decision="direct_samples_speed_up" if speed>=1.25 and bound[0]>1 else "unresolved",
                  world_ids=list(worlds),frames=len(ts),repetitions=reps,cells=side**3,exact=True,
                  full_scalars=side**3,read_scalars=3*side*PROBES+8*ANCHORS*4,
                  median_ms=(np.median(ts,axis=(0,1))/1e6).tolist(),paired_speedup=speed,
                  speedup_interval=bound.tolist(),frozen_checkpoints=checkpoints)
    if smoke:
        print(json.dumps(output))
        return
    out=Path(os.environ["STUDIO_JOB_DIR"])
    np.savez_compressed(out/"timings.npz",elapsed_ns=ts,decisions=decisions)
    (out/"result.json").write_text(json.dumps(output,indent=2)+"\n")


if __name__=="__main__":
    main()
