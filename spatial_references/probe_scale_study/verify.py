"""Check the two paired 192³ runs and reproduce the frozen scale decision."""
from hashlib import sha256
import json
from pathlib import Path
import runpy

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "results"
SHARED = runpy.run_path(str(HERE.parent / "stratified_regions_study/verify.py"))


def verify_tallies(arm, counts, shifts, side):
    assert counts.shape == (3, 8, 16, 3, 2, 6)
    assert shifts.shape == (3, 8, 16, 2, 2, 3)
    assert (counts[..., 2] == side**3).all() and (counts >= 0).all()
    for column in (1, 4):
        assert (counts[..., column] <= counts[..., 0]).all()
    for column in (3, 5):
        assert (counts[..., column] <= counts[..., 2]).all()
    assert (shifts >= 0).all() and (shifts < side//2).all()
    assert (shifts[..., 1, 0, :] == 0).all()
    assert (shifts[..., 1, 1, :] % 4 == 0).all() and (shifts[..., 1, 1, :] > 0).all()
    tally = counts[:, :, 8:].sum(axis=(0, 1, 2))
    np.testing.assert_array_equal(tally, arm["counts"])
    distance = np.abs(shifts[:, :, 8:, 0] - shifts[:, :, 8:, 1])
    distance = np.minimum(distance, side//2 - distance)
    for i, layout in enumerate(("aligned", "shifted")):
        observed = arm["boundary_accuracy"][layout]
        np.testing.assert_allclose([observed["exact"], observed["within_four"]],
                                   [(distance[..., i, :] == 0).mean(), (distance[..., i, :] <= 4).mean()])
        for j, policy in enumerate(("fixed", "inferred", "known")):
            valid, _, tested, retained, true_write, false_write = tally[j, i]
            recorded = arm["rates"][policy][layout]
            np.testing.assert_allclose([recorded["retention"], recorded["legitimate"], recorded["false_writes"]],
                                       [retained/tested, true_write/valid, false_write/tested])
    worlds = counts[:, :, 8:].sum(axis=(0, 2))
    difference, bounds = SHARED["delta"](worlds[:, 1], worlds[:, 0], 686132)
    SHARED["confirm"](arm["inferred_minus_fixed"], difference, bounds)
    return worlds, difference, bounds


def main():
    receipt = json.loads((R / "receipt.json").read_text())
    report = json.loads((R / "result.json").read_text())
    assert (receipt["job"], receipt["revision"], receipt["status"], receipt["exit"]) == (
        "spatial-fixed-probe-count-7077888", "a3392ce86577280a96aa94f6aad5e9519576c60b", "done", 0)
    side = report["side"]
    assert report["cells"] == side**3 == 7_077_888
    assert report["worlds"] == list(range(32, 40)) and report["transitions"] == 256
    assert report["probes_per_plane"] == 64 and report["sample_fraction_upper"] == 3*64/side**2
    checkpoints = report["frozen_checkpoints"]
    assert set(checkpoints) == {"101", "211", "307"}
    for key, digest in checkpoints.items():
        assert sha256((HERE.parent / "results" / f"cube-full-{key}.json").read_bytes()).hexdigest() == digest
    source = {}
    paired = {}
    for mode in ("full", "stratified"):
        with np.load(R / f"{mode}.npz") as raw:
            source[mode] = raw["counts"], raw["shifts"]
        paired[mode] = verify_tallies(report["modes"][mode], *source[mode], side)
    full, full_edges = source["full"]
    sampled, sampled_edges = source["stratified"]
    np.testing.assert_array_equal(full[:, :, :, (0, 2)], sampled[:, :, :, (0, 2)])
    np.testing.assert_array_equal(full_edges[..., 1, :, :], sampled_edges[..., 1, :, :])
    spatial, fixed_shift, bounds_fixed = paired["stratified"]
    compared, bounds = SHARED["delta"](spatial[:, 1], paired["full"][0][:, 1], 591340)
    SHARED["confirm"](report["stratified_minus_full"], compared, bounds)
    legit = min(q["legitimate"] for q in report["modes"]["stratified"]["rates"]["inferred"].values())
    criterion = (bounds[0, 1] >= -.02 and fixed_shift[1] >= .04 and bounds_fixed[0, 1] > 0 and legit >= .98)
    assert report["decision"] == ("fixed_probe_count_scales" if criterion else "unresolved")
    print(json.dumps(dict(**report, verifies_exact_fixed_known=True, paired_trials=24,
                          noninferiority_pass=bool(bounds[0, 1] >= -.02),
                          fixed_recovery_pass=bool(fixed_shift[1] >= .04 and bounds_fixed[0, 1] > 0),
                          legitimate_write_pass=bool(legit >= .98)), indent=2))


if __name__ == "__main__":
    main()
