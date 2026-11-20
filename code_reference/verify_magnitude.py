"""Check the precommitted nonlinear-control decision using paired byte windows."""
from pathlib import Path
import json
import numpy as np
from verify_working import compare
root=Path(__file__).resolve().parent/"results"

def audit():
    folder=root/"magnitude"
    record=json.loads((folder/"result.json").read_text());assert record["parameters"]==26830464
    metrics={}
    for split,label in (("val","validation"),("test","test")):
        current=np.load(folder/(label+".npz")); m=record["metrics"][split]
        np.testing.assert_allclose(current["scores"].mean(0)[:2]/512,[m["bpb"],m["accuracy"]])
        refs={"sealed":root/"bypass-main"/(label+".npz"),"duplicate":root/"working-main"/(label+".npz"),
              "fixed":root/"main"/("fixed-"+label+".npz"),"writable":root/"main"/("writable-"+label+".npz")}
        metrics[split]={k:compare(current,np.load(path),m[k]) for k,path in refs.items()}
    x=metrics["test"]
    if x["sealed"]["interval"][1]<.005 and x["duplicate"]["interval"][1]<0:
        outcome="final magnitudes recover most of the duplication penalty"
    elif x["sealed"]["interval"][0]>.005:
        outcome="early copy retains an advantage exceeding the registered margin"
    else:outcome="mechanism unresolved"
    return dict(contrasts=metrics,outcome=outcome)

if __name__=="__main__":print(json.dumps(audit(),indent=2))
