"""Recompute held-out loss and paired package intervals for all six fits."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from verify import grouped_difference
from verify_working import audit as prior_audit

def audit():
    root = ROOT/"results"
    earlier = prior_audit()
    results = {}
    repeat = json.loads((root/"working-repeat/result.json").read_text())
    for seed in ("101","211"):
        by_split={}
        for split,long_name in (("val","validation"),("test","test")):
            if seed=="101":
                files = {"readable":root/"main"/("fixed-"+long_name+".npz"),
                         "sealed":root/"bypass-main"/(long_name+".npz"),
                         "working":root/"working-main"/(long_name+".npz")}
            else:
                files = {"readable":root/"bypass-repeat"/("readable-"+split+".npz"),
                         "sealed":root/"bypass-repeat"/("sealed-"+split+".npz"),
                         "working":root/"working-repeat"/("working-"+split+".npz")}
            values = {k:np.load(p) for k,p in files.items()}
            for arm in ("readable","working"):
                for k in ("windows","groups"):
                    np.testing.assert_array_equal(values[arm][k],values["sealed"][k])
            score={k:float(v["scores"][:,0].mean()/512) for k,v in values.items()}
            accuracy={k:float(v["scores"][:,1].mean()/512) for k,v in values.items()}
            def contrast(a,b):
                return grouped_difference(values[a]["scores"][:,0],values[b]["scores"][:,0],values[b]["groups"])
            a,b=contrast("sealed","readable"),contrast("working","sealed")
            if seed=="211":
                target=repeat["working_minus_sealed"][split]
                np.testing.assert_allclose([b["delta"],*b["interval"]],[target["delta"],*target["interval"]])
                np.testing.assert_allclose(score["working"],repeat["arms"]["working"][split+"_bpb"])
                assert repeat["arms"]["working"]["parameters"]==26830464
                np.testing.assert_allclose(a["interval"],earlier["repeat_"+split]["interval"])
            else:
                np.testing.assert_allclose(b["interval"],earlier["main"][split+"_sealed"]["interval"])
            by_split[split]=dict(bits_per_byte=score,accuracy=accuracy,sealed_minus_readable=a,working_minus_sealed=b)
        results[seed]=by_split
    sources=json.loads((root/"main/corpus.json").read_text())["files"]
    group_sets=[{file["group"] for file in sources[split]} for split in ("train","val","test")]
    assert all(not (group_sets[a]&group_sets[b]) for a,b in ((0,1),(1,2),(0,2)))
    return results

if __name__=="__main__":
    print(json.dumps(audit(),indent=2))
