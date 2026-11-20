"""Audit the sub-one-percent 192³ estimator against its archived dense-axis baseline."""
from hashlib import sha256
import json
from pathlib import Path
import runpy

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "results"
BASE = HERE.parent / "probe_scale_study"
CHECK = runpy.run_path(str(BASE / "verify.py"))


def main():
    r = json.loads((R / "result.json").read_text())
    receipt = json.loads((R / "receipt.json").read_text())
    assert (receipt["job"], receipt["revision"], receipt["status"], receipt["exit"]) == (
        "spatial-anchored-witness-routing-7077888", "813daccd502125d79703bff8568755a65bdbade5", "done", 0)
    old = BASE / "results/stratified.npz"
    known = json.loads((BASE / "results/result.json").read_text())
    assert sha256(old.read_bytes()).hexdigest() == r["comparator_sha256"]["spatial-fixed-probe-count-7077888"]
    assert r["frozen_checkpoints"] == known["frozen_checkpoints"]
    for seed, digest in r["frozen_checkpoints"].items():
        assert sha256((HERE.parent / "results" / f"cube-full-{seed}.json").read_bytes()).hexdigest() == digest
    side = 192
    assert r["cells"] == side**3 and r["worlds"] == list(range(32,40))
    assert r["transitions"] == 256 and r["probes_per_plane"] == 64 and r["anchors_per_region"] == 1024
    assert r["sample_fraction_upper"] == 3*64/side**2 + 4*8*1024/side**3 < .01
    with np.load(R / "worlds.npz") as sparse, np.load(old) as dense:
        old_c, new_c = dense["counts"], sparse["counts"]
        old_s, new_s = dense["shifts"], sparse["shifts"]
    np.testing.assert_array_equal(new_s,old_s)
    np.testing.assert_array_equal(new_c[:,:,:,(0,2)],old_c[:,:,:,(0,2)])
    sampled, fixed, fixed_bounds = CHECK["verify_tallies"](r, new_c, new_s, side)
    benchmark, _, _ = CHECK["verify_tallies"](known["modes"]["stratified"],old_c,old_s,side)
    helper = CHECK["SHARED"]
    diff, spread = helper["delta"](sampled[:,1], benchmark[:,1], 591340)
    helper["confirm"](r["paired_minus_dense"], diff, spread)
    accepts = min(x["legitimate"] for x in r["rates"]["inferred"].values())
    passes = spread[0].min() >= -.01 and fixed[1] >= .04 and fixed_bounds[0,1] > 0 and accepts >= .98
    assert r["decision"] == ("dedicated_probes_scale" if passes else "unresolved")
    print(json.dumps(dict(**r, verified_equal_boundaries=True, verified_equal_fixed_known=True,
                          dense_orientation=known["modes"]["stratified"]["rates"]["inferred"],
                          full_field=known["modes"]["full"]["rates"]["inferred"]),indent=2))


if __name__=="__main__":
    main()
