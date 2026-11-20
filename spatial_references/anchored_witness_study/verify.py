"""Recheck dedicated witness sampling against its two archived comparison arms."""
from hashlib import sha256
import json
from pathlib import Path
import runpy

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "results"
DENSE = HERE.parent / "stratified_regions_study/results"
REUSE = HERE.parent / "sampled_witness_study/results"
STATS = runpy.run_path(str(DENSE.parent / "verify.py"))


def main():
    report = json.loads((R / "result.json").read_text())
    receipt = json.loads((R / "receipt.json").read_text())
    assert (receipt["job"], receipt["revision"], receipt["status"], receipt["exit"]) == (
        "spatial-anchored-witness-routing-884736", "9cf49311e47fd8a28ca3b7eb08dd5078cd835a15", "done", 0)
    assert report["cells"] == 96**3 and report["worlds"] == 8 and report["transitions"] == 256
    assert report["anchors_per_region"] == 1024 and report["probes_per_plane"] == 64
    assert report["sample_fraction_upper"] == 3*64/96**2 + 4*8*1024/96**3
    for seed, digest in report["frozen_checkpoints"].items():
        assert sha256((HERE.parent / "results" / f"cube-full-{seed}.json").read_bytes()).hexdigest() == digest
    with np.load(R / "worlds.npz") as f:
        anchored, shifts = f["counts"], f["shifts"]
    assert anchored.shape == (3, 8, 16, 3, 2, 6) and (anchored[..., 2] == 96**3).all()
    tally = anchored[:, :, 8:].sum(axis=(0, 1, 2))
    np.testing.assert_array_equal(tally, report["counts"])
    for i, mode in enumerate(("fixed", "inferred", "known")):
        for j, layout in enumerate(("aligned", "shifted")):
            n, _, m, correct, legitimate, false = tally[i, j]
            r = report["rates"][mode][layout]
            np.testing.assert_allclose([r["retention"], r["legitimate"], r["false_writes"]],
                                       [correct/m, legitimate/n, false/m])
    a = anchored[:, :, 8:].sum(axis=(0, 2))
    tests = {}
    for name, directory, key in (("dense", DENSE, "spatial-stratified-region-probes"),
                                 ("shared", REUSE, "spatial-sampled-witness-routing-884736")):
        old = directory / "worlds.npz"
        assert sha256(old.read_bytes()).hexdigest() == report["comparator_sha256"][key]
        source = json.loads((directory / "result.json").read_text())
        assert source["frozen_checkpoints"] == report["frozen_checkpoints"]
        assert source["boundary_accuracy"] == report["boundary_accuracy"]
        with np.load(old) as f:
            matched = f["counts"]
            np.testing.assert_array_equal(shifts, f["shifts"])
        np.testing.assert_array_equal(anchored[:, :, :, (0, 2)], matched[:, :, :, (0, 2)])
        b = matched[:, :, 8:].sum(axis=(0, 2))
        point, bounds = STATS["delta"](a[:, 1], b[:, 1], 591340)
        STATS["confirm"](report[f"paired_minus_{name}"], point, bounds)
        tests[name] = point, bounds
    gain, safety = STATS["delta"](a[:, 1], a[:, 0], 686132)
    STATS["confirm"](report["inferred_minus_fixed"], gain, safety)
    effect, spread = tests["shared"]
    passed = (tests["dense"][1][0].min() >= -.01 and gain[1] >= .04 and safety[0, 1] > 0
              and effect[1] >= .02 and spread[0, 1] > 0 and (tally[1, :, 4]/tally[1, :, 0]).min() >= .98)
    assert report["decision"] == ("dedicated_probes_restore_routing" if passed else "unresolved")
    print(json.dumps(dict(**report, archived_controls_and_boundaries_match=True), indent=2))


if __name__ == "__main__":
    main()
