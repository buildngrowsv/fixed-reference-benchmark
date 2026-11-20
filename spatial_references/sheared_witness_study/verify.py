"""Reconstruct the separated shear and retained-word outcomes."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
root = here / "results"
receipt = json.loads((root / "receipt.json").read_text())
assert (receipt["job"], receipt["revision"], receipt["status"], receipt["exit"]) == (
    "spatial-sheared-witness-7077888", "2315111632a69a9e901e01d9f4d796ce401045eb", "done", 0)
r = json.loads((root / "result.json").read_text())
weight = here.parent / "results/cube-full-101.json"
assert sha256(weight.read_bytes()).hexdigest() == r["checkpoint_sha256"]
with np.load(root / "sheared.npz") as archive:
    counts, offsets, shear = (archive[k] for k in ("counts","shifts","fitted_shears"))
assert counts.shape == (8,16,3,2,6) and offsets.shape == (8,16,2,2,3) and shear.shape == (8,16,2)
assert r["cells"] == 192**3 and (counts[..., 2] == r["cells"]).all()
assert r["worlds"] == list(range(64,72)) and (offsets[...,1,0,:] == 0).all()
assert (offsets[:,:,1] == offsets[:,:1,1]).all()
np.testing.assert_array_equal(r["true_shears"], np.random.default_rng(223247).permutation([-1]*4+[1]*4))
correct_shear = shear[:,8:,1] == np.asarray(r["true_shears"])[:,None]
assert correct_shear.mean() == r["shifted_shear_accuracy"]
truth = (offsets[:,8:,0] == offsets[:,8:,1])
for j, k in enumerate(("aligned","shifted")):
    assert truth[...,j,:].mean() == r["boundary_accuracy"][k]["exact"]
late=counts[:,8:].sum(1)
pooled=late.sum(0)
np.testing.assert_array_equal(pooled,r["counts"])
np.testing.assert_array_equal(counts[...,4],np.broadcast_to(counts[:,:,:1,:,4],counts[...,4].shape))
for i, mode in enumerate(("fixed","inferred","known")):
    for j, stimulus in enumerate(("aligned","shifted")):
        a,_,b,c,d,f = pooled[i,j]
        reported=r["rates"][mode][stimulus]
        np.testing.assert_allclose([c/b,d/a,f/b],[reported[k] for k in ("retention","legitimate","false_writes")])
rates=late[...,3]/late[...,2]
draws=np.random.default_rng(562218).integers(8,size=(5000,8))
def compare(a,b):
    delta=rates[:,a]-rates[:,b]
    return dict(mean=delta.mean(0).tolist(),interval=np.quantile(delta[draws].mean(1),[.025,.975],axis=0).T.tolist())
for name, ctrl in (("adaptive_minus_axis_aligned",0),("adaptive_minus_supplied",2)):
    reconstructed=compare(1,ctrl)
    for key,v in reconstructed.items():
        np.testing.assert_allclose(v,r[name][key])
sa=r["adaptive_minus_axis_aligned"]
sb=r["adaptive_minus_supplied"]
pass_= (correct_shear.mean()>=.8 and sa["mean"][1]>=.02 and sa["interval"][1][0]>0 and sb["interval"][1][0]>=-.015
        and sa["interval"][0][0]>=-.005 and (pooled[...,4]/pooled[...,0]).min()>=.98)
assert r["decision"] == ("sparse_shear_adaptation_recovers_words" if pass_ else "unresolved")
print(json.dumps(dict(retention=r["rates"], shifted_correct_shears=int(correct_shear.sum()),
                      shifted_exact_offset_vectors=int(truth[...,1,:].all(-1).sum()), late_frames=64,
                      supplied_minus_axis=compare(2,0),
                      adaptive_minus_axis=sa,adaptive_minus_supplied=sb),indent=2))
