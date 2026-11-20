"""Verify the matched controller runs and recalculate coverage comparisons."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RECORDS = {
    "full": (ROOT / "inferred_regions_study/results", "spatial-inferred-regions-884736",
             "198240fb667f2c1d2c6f5b70a9a28929d825b809"),
    "random": (ROOT / "sparse_regions_study/results", "spatial-sparse-inferred-regions-884736",
               "8a5f1d510a943610388c95a23d57f56ab7deae1b"),
    "stratified": (HERE / "results", "spatial-stratified-region-probes",
                   "24739b3275db87b8e1d05a0a9b7cc861381f0e73"),
}


def read(path, name, revision):
    receipt = json.loads((path / "receipt.json").read_text())
    assert (receipt["job"], receipt["revision"], receipt["status"], receipt["exit"]) == (
        name, revision, "done", 0)
    report = json.loads((path / "result.json").read_text())
    with np.load(path / "worlds.npz") as z:
        counts, shifts = z["counts"], z["shifts"]
    assert counts.shape == (3, 8, 16, 3, 2, 6)
    assert shifts.shape == (3, 8, 16, 2, 2, 3)
    return report, counts, shifts


def rate(tally):
    return tally[..., 3] / tally[..., 2]


def delta(new, prior, seed):
    point = rate(new.sum(axis=0)) - rate(prior.sum(axis=0))
    rng = np.random.default_rng(seed)
    indices = rng.integers(8, size=(4000, 8))
    difference = rate(new[indices].sum(axis=1)) - rate(prior[indices].sum(axis=1))
    return point, np.percentile(difference, [2.5, 97.5], axis=0)


def confirm(report, means, bounds):
    for i, field in enumerate(("aligned", "shifted")):
        np.testing.assert_allclose([report[field]["mean"], *report[field]["interval"]],
                                   [means[i], *bounds[:, i]])


def main():
    data = {mode: read(*identity) for mode, identity in RECORDS.items()}
    ref, counts, layout = data["stratified"]
    for seed, digest in ref["frozen_checkpoints"].items():
        assert sha256((ROOT / "results" / f"cube-full-{seed}.json").read_bytes()).hexdigest() == digest
    panel = {}
    for mode, (summary, tally, shifts) in data.items():
        assert summary["frozen_checkpoints"] == ref["frozen_checkpoints"]
        np.testing.assert_array_equal(tally[:, :, :, (0, 2)], counts[:, :, :, (0, 2)])
        np.testing.assert_array_equal(shifts[:, :, :, 1], layout[:, :, :, 1])
        assert (tally[..., 2] == 96**3).all()
        totals = tally[:, :, 8:].sum(axis=(0, 1, 2))
        np.testing.assert_array_equal(totals, summary["counts"])
        err = np.abs(shifts[:, :, 8:, 0] - shifts[:, :, 8:, 1])
        err = np.minimum(err, 48 - err)
        for stimulus, i in (("aligned", 0), ("shifted", 1)):
            accuracy = summary["boundary_accuracy"][stimulus]
            assert accuracy == dict(exact=float((err[..., i, :] == 0).mean()),
                                    within_four=float((err[..., i, :] <= 4).mean()))
            for arm, row in zip(("fixed", "inferred", "known"), totals[:, i], strict=True):
                results = summary["rates"][arm][stimulus]
                np.testing.assert_allclose([row[3]/row[2], row[4]/row[0], row[5]/row[2]],
                                           [results["retention"], results["legitimate"], results["false_writes"]])
        panel[mode] = tally[:, :, 8:].sum(axis=(0, 2))
    current = panel["stratified"][:, 1]
    to_fix, bounds_fix = delta(current, panel["stratified"][:, 0], 686132)
    to_full, bounds_full = delta(current, panel["full"][:, 1], 591340)
    to_random, bounds_random = delta(current, panel["random"][:, 1], 591340)
    confirm(ref["inferred_minus_fixed"], to_fix, bounds_fix)
    confirm(ref["sparse_minus_full"], to_full, bounds_full)
    confirm(ref["stratified_minus_random"], to_random, bounds_random)
    assert ref["strata_per_axis"]**2 == ref["probes_per_plane"] == 64
    assert ref["sampled_upper_fraction"] == 3*64/96**2
    floor = min(v["legitimate"] for v in ref["rates"]["inferred"].values())
    enough = bounds_full[0, 1] >= -.02 and to_fix[1] >= .04 and bounds_fix[0, 1] > 0 and floor >= .98
    placement = to_random[1] >= .005 and bounds_random[0, 1] > 0
    assert ref["decision"] == ("distributed_probes_help" if enough and placement else "unresolved")
    print(json.dumps(dict(**ref, noninferiority_gate=bool(enough), placement_gate=bool(placement),
                          comparators={m: {"rates": d[0]["rates"]["inferred"],
                                           "boundary": d[0]["boundary_accuracy"]} for m, d in data.items()}),
                     indent=2))


if __name__ == "__main__":
    main()
