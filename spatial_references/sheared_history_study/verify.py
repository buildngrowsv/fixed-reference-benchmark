"""Reconstruct archived offset and retention comparisons from the integer counts."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

here=Path(__file__).resolve().parent
root=here/"results"
receipt=json.loads((root/"receipt.json").read_text())
assert (receipt["job"],receipt["revision"],receipt["status"],receipt["exit"])==(
    "spatial-sheared-offset-history-7077888","7406f3915ba56d3fadae35c4858ea82b5c3725c8","done",0)
r=json.loads((root/"result.json").read_text())
base=here.parent/"sheared_witness_study/results/sheared.npz"
weights=here.parent/"results/cube-full-101.json"
assert sha256(base.read_bytes()).hexdigest()==r["baseline_sha256"]
assert sha256(weights.read_bytes()).hexdigest()==r["checkpoint_sha256"]
with np.load(root/"pooled.npz") as old, np.load(base) as prior:
    t=old["counts"]
    offsets=old["offsets"]
    np.testing.assert_array_equal(t[:,:,0],prior["counts"][:,:,1])
    np.testing.assert_array_equal(t[:,:,2],prior["counts"][:,:,2])
    np.testing.assert_array_equal(old["shears"],prior["fitted_shears"])
    np.testing.assert_array_equal(offsets[:,:,1],prior["shifts"][:,:,1])
assert t.shape==(8,16,3,2,6) and offsets.shape==(8,16,2,2,3)
assert r["worlds"]==list(range(64,72)) and (t[...,2]==192**3).all()
exact=(offsets[:,8:,0,1]==offsets[:,8:,1,1]).all(-1)
assert exact.mean()==r["exact_tilted_offset_vectors"]
late=t[:,8:].sum(axis=1)
rate=late[...,3]/late[...,2]
np.testing.assert_allclose(rate.mean(axis=0),r["retention"])
draws=np.random.default_rng(562218).integers(8,size=(5000,8))
for key,reference in (("improvement",0),("versus_supplied",2)):
    delta=rate[:,1]-rate[:,reference]
    bounds=np.quantile(delta[draws].mean(1),[.025,.975],axis=0).T
    np.testing.assert_allclose(delta.mean(0),r[key]["mean"])
    np.testing.assert_allclose(bounds,r[key]["interval"])
gain,loss=r["improvement"],r["versus_supplied"]
passes=(exact.mean()>=.90 and gain["mean"][1]>=.01 and gain["interval"][1][0]>0
        and loss["interval"][1][0]>=-.0075 and gain["interval"][0][0]>=-.005)
assert r["decision"]==("history_repairs_tilted_offsets" if passes else "unresolved")
print(json.dumps(dict(decision=r["decision"],retention=rate.mean(0).tolist(),recovered_offsets=int(exact.sum()),
                      total_scored=exact.size,improvement=gain,versus_supplied=loss),indent=2))
