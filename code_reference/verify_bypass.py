"""Audit the sealed reference using the original held-out window records."""
import json
import sys
from pathlib import Path
import numpy as np
from verify import grouped_difference

def main(base,full):
    record=json.loads((base/"result.json").read_text())
    audit={"fraction_weights_saved":1-record["parameters"]/31992320}
    for split,name in (("validation","val"),("test","test")):
        v=np.load(base/(split+".npz"));s=v["scores"];m=record["metrics"][name]
        np.testing.assert_allclose([s[:,0].mean()/512,s[:,1].mean()/512],[m["bpb"],m["accuracy"]])
        pairs={}
        for arm in ("fixed","writable"):
            ref=np.load(full/(arm+"-"+split+".npz"))
            for k in ("groups","windows"):np.testing.assert_array_equal(v[k],ref[k])
            pair=grouped_difference(s[:,0],ref["scores"][:,0],v["groups"])
            np.testing.assert_allclose(pair["interval"],m[arm]["ci"])
            np.testing.assert_allclose(pair["delta"],m[arm]["delta"])
            pairs[arm]=pair
        audit[split]=pairs
    print(json.dumps(audit,indent=2))
if __name__=="__main__":main(*map(Path,sys.argv[1:]))
