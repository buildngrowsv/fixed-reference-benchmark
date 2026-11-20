"""Recompute the limited-observation comparison from both released runs."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
path = ROOT / "results"
rec = json.loads((path / "result.json").read_text())
receipt = json.loads((path / "receipt.json").read_text())
assert (receipt["job"], receipt["revision"], receipt["status"], receipt["exit"]) == (
    "spatial-sparse-inferred-regions-884736", "8a5f1d510a943610388c95a23d57f56ab7deae1b", "done", 0)
for seed, digest in rec["frozen_checkpoints"].items():
    assert sha256((ROOT.parent / "results" / f"cube-full-{seed}.json").read_bytes()).hexdigest() == digest
previous = ROOT.parent / "inferred_regions_study" / "results"
old = json.loads((previous / "result.json").read_text())
assert rec["frozen_checkpoints"] == old["frozen_checkpoints"]
with np.load(previous / "worlds.npz") as orig, np.load(path / "worlds.npz") as data:
    control, limited = orig["counts"], data["counts"]
    original_layout, sampled_layout = orig["shifts"], data["shifts"]
assert control.shape == limited.shape == (3,8,16,3,2,6)
np.testing.assert_array_equal(control[:,:,:,(0,2)], limited[:,:,:,(0,2)])
np.testing.assert_array_equal(original_layout[:,:,:,1], sampled_layout[:,:,:,1])
assert (limited[...,2] == 96**3).all() and (limited[...,3] <= limited[...,2]).all()
assert rec["probes_per_plane"] == 64 and rec["sampled_upper_fraction"] == 3*64/96**2
total = limited[:,:,8:].sum(axis=(0,1,2))
np.testing.assert_array_equal(total, rec["counts"])
p = limited[:,:,8:].sum(axis=(0,2))
prior = control[:,:,8:].sum(axis=(0,2))
deltas = []
rng = np.random.default_rng(591340)
for worlds in rng.integers(8,size=(4000,8)):
    x, y = p[worlds].sum(axis=0), prior[worlds].sum(axis=0)
    deltas.append(x[1,:,3]/x[1,:,2] - y[1,:,3]/y[1,:,2])
full_total = prior.sum(axis=0)
truth = total[1,:,3]/total[1,:,2] - full_total[1,:,3]/full_total[1,:,2]
confidence = np.percentile(deltas,[2.5,97.5],axis=0)
errors = np.abs(sampled_layout[:,:,8:,0] - sampled_layout[:,:,8:,1])
errors = np.minimum(errors,48-errors)
for i, stimulus in enumerate(("aligned","shifted")):
    v = rec["sparse_minus_full"][stimulus]
    np.testing.assert_allclose([truth[i],*confidence[:,i]], [v["mean"],*v["interval"]])
    assert rec["boundary_accuracy"][stimulus]["exact"] == (errors[...,i,:]==0).mean()
    assert rec["boundary_accuracy"][stimulus]["within_four"] == (errors[...,i,:]<=4).mean()
    for j, arm in enumerate(("fixed", "inferred", "known")):
        x=total[j,i]
        np.testing.assert_allclose(list(rec["rates"][arm][stimulus].values()),[x[3]/x[2],x[4]/x[0],x[5]/x[2]])
ret = rec["inferred_minus_fixed"]["shifted"]
assert rec["decision"] == ("sparse_layout_sufficient" if confidence[0,1]>=-.02 and ret["mean"]>=.04
                          and ret["interval"][0]>0 and (total[1,:,4]/total[1,:,0]>=.98).all() else "unresolved")
print(json.dumps(dict(**rec,full_observation=old["rates"]["inferred"],
                      full_boundary_accuracy=old["boundary_accuracy"]),indent=2))
