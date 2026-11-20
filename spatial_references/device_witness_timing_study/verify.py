"""Recompute exact parity and the paired world-bootstrap timing comparison."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
R=HERE/"results"


def main():
    result=json.loads((R/"result.json").read_text())
    receipt=json.loads((R/"receipt.json").read_text())
    assert (receipt["job"],receipt["revision"],receipt["status"],receipt["exit"]) == (
        "spatial-direct-witness-observation-timing","1b375f271a66ad1307b40511acd1cf5fd8568be8","done",0)
    assert result["world_ids"]==list(range(32,40)) and result["frames"]==48 and result["repetitions"]==12
    for name,digest in result["frozen_checkpoints"].items():
        assert sha256((HERE.parent/"results"/f"cube-full-{name}.json").read_bytes()).hexdigest()==digest
    with np.load(R/"timings.npz") as f:
        raw,decisions=f["elapsed_ns"],f["decisions"]
    assert raw.shape==(48,12,2) and (raw>0).all() and decisions.shape==(48,2,11)
    np.testing.assert_array_equal(decisions[:,0],decisions[:,1])
    assert (decisions[:,:,:3]>=0).all() and (decisions[:,:,:3]<96).all()
    assert (decisions[:,:,3:]>=0).all() and (decisions[:,:,3:]<3).all()
    assert result["cells"]==result["full_scalars"]==192**3 and result["read_scalars"]==3*192*64+4*8*1024
    ratios=np.median(raw[...,0],axis=1)/np.median(raw[...,1],axis=1)
    worlds=np.median(ratios.reshape(3,8,2),axis=(0,2))
    rng=np.random.default_rng(591277)
    draws=rng.integers(8,size=(5000,8))
    interval=np.percentile(np.median(worlds[draws],axis=1),[2.5,97.5])
    speed=float(np.median(worlds))
    np.testing.assert_allclose([speed,*interval],[result["paired_speedup"],*result["speedup_interval"]])
    np.testing.assert_allclose(np.median(raw,axis=(0,1))/1e6,result["median_ms"])
    assert result["exact"] and result["decision"]==("direct_samples_speed_up" if speed>=1.25 and interval[0]>1 else "unresolved")
    print(json.dumps(dict(**result,world_ratios=worlds.tolist(),minimum_frame_speedup=float(ratios.min())),
                     indent=2))


if __name__=="__main__":
    main()
