"""Verify fresh-program outcomes and the paired continuation contrasts."""
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent/"results"
sys.path.insert(0,str(HERE.parents[1]/"reference_lm"))
from programs import batch


def gap(a,b):
    difference=(a-b)[:,[1,4]]
    denom=a[:,[0,3]]
    rng=np.random.default_rng(192841)
    draws=[]
    for _ in range(40):
        idx=rng.integers(len(a),size=(100,len(a)))
        draws.extend((difference[idx].sum(1)/denom[idx].sum(1)).mean(1))
    lo,hi=np.quantile(draws,[.025,.975])
    return dict(delta=float((difference.sum(0)/denom.sum(0)).mean()),interval=[float(lo),float(hi)])


def main():
    tally=[]
    for i in range(32):
        _,y,op,changed,guess=batch([427361,i])
        group=[]
        for kind in (2,3):
            mask=changed&(op==kind)
            group.extend([mask.sum(1),((guess==y)&mask).sum(1)])
        tally.extend(np.column_stack(group))
    data={}
    performance={}
    for fit,folder in (("quarter","large-quarter-rate"),("original","large-original-routed")):
        root=RESULTS/folder
        summary=json.loads((root/"result.json").read_text())
        assert summary["target"]==4096 and summary["resumed_from_update"]==512
        for arm,details in summary["arms"].items():
            assert details["status"]=="complete"
            a=np.load(root/(arm+"-counts.npy"))
            np.testing.assert_array_equal(a[:,[0,2,3,5]],tally)
            assert np.all(a>=0) and np.all(a[:,[1,4]]<=a[:,[0,3]])
            z=a.sum(0);correct=z[[1,4]]/z[[0,3]]
            np.testing.assert_allclose(correct,details["accuracy"])
            name=fit+"_"+arm
            performance[name]=float(correct.mean())
            data[name]=a
    matched=gap(data["quarter_rotated"],data["quarter_dense"])
    summary=json.loads((RESULTS/"large-quarter-rate/result.json").read_text())
    np.testing.assert_allclose([matched["delta"],*matched["interval"]],[summary["gap"],*summary["interval"]])
    assert (matched["delta"]>=.10 and matched["interval"][0]>0)==summary["recovered_benefit"]
    report=dict(programs=len(tally),query_counts=np.array(tally).sum(0)[[0,2]].tolist(),balanced=performance,
                paired_route_advantage=matched,quarter_minus_original_routed=gap(data["quarter_rotated"],data["original_rotated"]))
    baseline = RESULTS / "large-budget-baseline"
    baseline_summary = json.loads((baseline / "result.json").read_text())
    assert baseline_summary["panel_seed"] == 427361 and baseline_summary["updates"] == 512
    reports = {}
    with np.load(baseline / "counts.npz") as archived:
        for arm in ("rotated", "dense"):
            original = archived[arm]
            np.testing.assert_array_equal(original[:, [0, 2, 3, 5]], tally)
            totals = original.sum(0)
            accuracy = totals[[1, 4]] / totals[[0, 3]]
            np.testing.assert_allclose(accuracy, baseline_summary["accuracy"][arm])
            reports[arm] = dict(before=float(accuracy.mean()),
                                after_minus_before=gap(data["quarter_" + arm], original))
    report["matched_before_and_after"] = reports
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
