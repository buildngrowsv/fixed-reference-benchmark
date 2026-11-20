"""Check each compact-arm comparison against the archived package and byte windows."""
from pathlib import Path
import json
import numpy as np
from verify import grouped_difference

ROOT = Path(__file__).resolve().parent/"results"


def compare(left, right, expected):
    for key in ("groups","windows"):
        np.testing.assert_array_equal(left[key],right[key])
    result = grouped_difference(left["scores"][:,0],right["scores"][:,0],left["groups"])
    np.testing.assert_allclose(result["interval"],expected["ci"] if "ci" in expected else expected["interval"])
    np.testing.assert_allclose(result["delta"],expected["delta"])
    return result


def audit():
    report = {}
    for phase,name in (("screen","mid"),("main","main")):
        folder = ROOT/("working-"+phase)
        record = json.loads((folder/"result.json").read_text())
        assert record["parameters"] == 26830464
        results = {}
        for split,m in record["metrics"].items():
            label = "validation" if split == "val" else split
            left = np.load(folder/(label+".npz"))
            totals = left["scores"].mean(axis=0)[:2]/512
            np.testing.assert_allclose(totals,[m["bpb"],m["accuracy"]])
            for arm in ("fixed","writable","sealed"):
                parent = ROOT/name/(arm+"-"+label+".npz") if arm!="sealed" else ROOT/("bypass-"+phase)/(label+".npz")
                results[split+"_"+arm] = compare(left,np.load(parent),m[arm])
        report[phase] = results
    folder = ROOT/"bypass-repeat"
    repeat = json.loads((folder/"result.json").read_text())
    for split,label in (("val","validation"),("test","test")):
        left,right = (np.load(folder/(arm+"-"+split+".npz")) for arm in ("sealed","readable"))
        for k in ("groups","windows"):
            np.testing.assert_array_equal(left[k],np.load(ROOT/"bypass-main"/(label+".npz"))[k])
        values = [repeat["arms"][arm][split+"_bpb"] for arm in ("sealed","readable")]
        np.testing.assert_allclose(values,[v["scores"][:,0].mean()/512 for v in (left,right)])
        report["repeat_"+split] = compare(left,right,dict(delta=values[0]-values[1],interval=repeat["intervals"][split]))
    assert repeat["qualifies"] == bool(repeat["arms"]["sealed"]["parameters"]<=.85*repeat["arms"]["readable"]["parameters"] and report["repeat_test"]["interval"][1]<.02)
    return report


if __name__=="__main__":
    print(json.dumps(audit(),indent=2))
