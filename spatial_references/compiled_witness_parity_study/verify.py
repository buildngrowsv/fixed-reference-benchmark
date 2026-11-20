"""Audit the compiled observer outputs and exact recurrent-simulation parity."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
R=HERE/"results"
OLDER=HERE.parent/"anchored_witness_scale_study"/"results"/"worlds.npz"


def main():
    r=json.loads((R/"result.json").read_text())
    receipt=json.loads((R/"receipt.json").read_text())
    assert (receipt["job"],receipt["revision"],receipt["status"],receipt["exit"]) == (
        "spatial-compiled-witness-parity-7077888","9dad3fd31226a763be3c11ae00c4453a30522463","done",0)
    seed=r["checkpoint"]
    assert seed==101 and r["worlds"]==list(range(32,40))
    weights=HERE.parent/"results"/f"cube-full-{seed}.json"
    assert sha256(weights.read_bytes()).hexdigest()==r["checkpoint_digest"]
    assert sha256(OLDER.read_bytes()).hexdigest()==r["source_archive_digest"]
    with np.load(R/"parity.npz") as x, np.load(OLDER) as reference:
        c, edges, outputs=x["counts"],x["shifts"],x["decisions"]
        old_counts,old_edges=reference["counts"][0],reference["shifts"][0]
    assert c.shape==(8,16,3,2,6) and edges.shape==(8,16,2,2,3)
    assert r["cells"]==192**3 and r["transitions"]==256 and (c[...,2]==r["cells"]).all()
    assert outputs.shape==(8,64,11) and r["decisions"]==np.prod(outputs.shape[:2])==512
    assert r["sampled_scalars"]==3*64*192+4*8*1024 and r["observer_output_scalars"]==11
    assert (outputs[:,:,:3]>=0).all() and (outputs[:,:,:3]<96).all()
    assert (outputs[:,:,3:]>=0).all() and (outputs[:,:,3:]<3).all()
    np.testing.assert_array_equal(outputs.reshape(8,16,2,2,11)[:,:,1,:,:3],edges[:,:,0])
    matched=np.array_equal(c,old_counts) and np.array_equal(edges,old_edges)
    assert bool(matched)==r["counts_and_shifts_exact"]
    assert r["decision"]==("compiled_observer_exact_parity" if matched else "unresolved")
    new=c[:,8:,1,1,:].sum(axis=1)
    prior=old_counts[:,8:,1,1,:].sum(axis=1)
    print(json.dumps(dict(**r,matched_tallies=int((c==old_counts).sum()),
                          matched_offset_coordinates=int((edges==old_edges).sum()),
                          shifted_retention_by_world=(new[:,3]/new[:,2]).tolist(),
                          reference_retention_by_world=(prior[:,3]/prior[:,2]).tolist()),indent=2))


if __name__=="__main__":
    main()
