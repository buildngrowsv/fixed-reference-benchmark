"""Verify sparse witness routing against the matched dense-orientation archive."""
from hashlib import sha256
import json
from pathlib import Path
import runpy

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "results"
BASE = HERE.parent / "stratified_regions_study/results"
STATS = runpy.run_path(str(BASE.parent / "verify.py"))
MODES = ("fixed", "inferred", "known")
LAYOUTS = ("aligned", "shifted")


def main():
    receipt = json.loads((R / "receipt.json").read_text())
    report = json.loads((R / "result.json").read_text())
    assert (receipt["job"], receipt["revision"], receipt["status"], receipt["exit"]) == (
        "spatial-sampled-witness-routing-884736", "5ef2af27507aa921f79c95a22bef7dcee670ccb0", "done", 0)
    archive = BASE / "worlds.npz"
    assert sha256(archive.read_bytes()).hexdigest() == report["matched_archive_sha256"]
    assert report["cells"] == 96**3 and report["worlds"] == 8 and report["transitions"] == 256
    assert report["probes_per_plane"] == 64 and report["sample_fraction_upper"] == 3*64/96**2
    dense = json.loads((BASE / "result.json").read_text())
    assert report["frozen_checkpoints"] == dense["frozen_checkpoints"]
    for key, digest in report["frozen_checkpoints"].items():
        assert sha256((HERE.parent / "results" / f"cube-full-{key}.json").read_bytes()).hexdigest() == digest
    with np.load(archive) as original, np.load(R / "worlds.npz") as current:
        full, shared = original["counts"], current["counts"]
        np.testing.assert_array_equal(original["shifts"], current["shifts"])
    assert report["boundary_accuracy"] == dense["boundary_accuracy"]
    assert full.shape == shared.shape == (3, 8, 16, 3, 2, 6)
    np.testing.assert_array_equal(full[:, :, :, (0, 2)], shared[:, :, :, (0, 2)])
    assert (shared[..., 2] == 96**3).all()
    total = shared[:, :, 8:].sum(axis=(0, 1, 2))
    np.testing.assert_array_equal(total, report["counts"])
    for p, mode in enumerate(MODES):
        for k, layout in enumerate(LAYOUTS):
            valid, _, tested, retained, true, false = total[p, k]
            results = report["rates"][mode][layout]
            np.testing.assert_allclose([retained/tested, true/valid, false/tested],
                                       [results["retention"], results["legitimate"], results["false_writes"]])
    a, b = (v[:, :, 8:].sum(axis=(0, 2)) for v in (shared, full))
    diff, bounds = STATS["delta"](a[:, 1], b[:, 1], 591340)
    STATS["confirm"](report["paired_minus_full_routing"], diff, bounds)
    gain, fixed_bounds = STATS["delta"](a[:, 1], a[:, 0], 686132)
    STATS["confirm"](report["inferred_minus_fixed"], gain, fixed_bounds)
    accepted = (total[1, :, 4]/total[1, :, 0]).min()
    criteria = bounds[0].min() >= -.01 and gain[1] >= .04 and fixed_bounds[0, 1] > 0 and accepted >= .98
    assert report["decision"] == ("same_probes_support_routing" if criteria else "unresolved")
    print(json.dumps(dict(**report, dense_routing=dense["rates"]["inferred"],
                          matching_boundaries=True, matching_fixed_and_known=True), indent=2))


if __name__ == "__main__":
    main()
