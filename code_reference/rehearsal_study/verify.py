"""Audit replay and the equal-size late-copy control against identical test windows."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "results"


def external(a, b):
    np.testing.assert_array_equal(a[:, :3], b[:, :3])
    rng = np.random.default_rng(282471)
    means, draws = [], []
    for project in range(6):
        use = a[:, 0] == project
        groups = a[use, 1]
        delta = a[use, 3] - b[use, 3]
        ids = np.unique(groups)
        totals = np.array([sum(delta[groups == i]) for i in ids])
        sizes = np.array([(groups == i).sum() for i in ids])
        sample = rng.choice(len(ids), (3000, len(ids)))
        means.append(delta.mean())
        draws.append(totals[sample].sum(1) / sizes[sample].sum(1))
    return [*means, np.mean(means), *np.quantile(np.mean(draws, axis=0), [.025, .975])]


def main():
    reference = np.load(ROOT / "ecosystem/projects.npz")
    working = np.load(ROOT / "ecosystem-working/projects.npz")
    replay = np.load(ROOT / "rehearsal/external.npz")["scores"]
    report = json.loads((ROOT / "rehearsal/result.json").read_text())
    late = json.loads((ROOT / "ecosystem-working/result.json").read_text())
    before, after = reference["sealed_before"], reference["sealed_after"]
    threshold = before.copy()
    threshold[:, 3] = .25 * before[:, 3] + .75 * after[:, 3]
    for a, b, stated in [(replay, threshold, report["external_vs_75pct_gain"]),
                         (replay, after, report["external_vs_no_replay"]),
                         (working["after"], after, late["working_minus_sealed"]["after"])]:
        np.testing.assert_allclose(external(a, b), [*stated["project_differences"], stated["macro"], *stated["interval"]])
    a, b, c = [np.load(ROOT / f) for f in ("bypass-main/test.npz", "ecosystem/sealed-retention.npz", "rehearsal/cpython.npz")]
    for key in ("groups", "windows"):
        np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_array_equal(a[key], c[key])
    delta = c["scores"][:, 0] - (a["scores"][:, 0] + b["scores"][:, 0])/2
    groups = a["groups"]
    ids = np.unique(groups)
    sums = np.array([delta[groups == i].sum() for i in ids])
    sizes = np.array([(groups == i).sum() for i in ids])
    sample = np.random.default_rng(9651).choice(len(ids), (5000, len(ids)))
    ci = np.quantile(sums[sample].sum(1) / (512*sizes[sample].sum(1)), [.025, .975])
    np.testing.assert_allclose([delta.mean()/512, *ci], [report["cpython_vs_half_penalty"]["delta"], *report["cpython_vs_half_penalty"]["interval"]])
    values = [x["scores"][:, 0].mean()/512 for x in (a, b, c)]
    assert (report["external_vs_75pct_gain"]["interval"][1] < 0 and ci[1] < 0) == report["supported"]
    print(json.dumps(dict(external_windows=len(replay), cpython_windows=len(delta), packages=len(ids), cpython_bpb=values,
          external_bpb=[x[:, 3].mean() for x in (before, after, replay)],
          fraction_of_forgetting_removed=(values[1]-values[2])/(values[1]-values[0])), indent=2))


if __name__ == "__main__":
    main()
