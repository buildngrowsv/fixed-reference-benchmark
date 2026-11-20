"""Audit register scores using per-program counts and an independent bootstrap."""
import sys,json
from pathlib import Path
import numpy as np

def main(root):
    report={}
    for stem in ("validation","test"):
        a=np.load(root/(stem+".npz"))
        metrics=json.loads((root/"result.json").read_text())["metrics"][stem]
        comparison={}
        for k in a.files:
            v=a[k];c=v.sum(0)
            np.testing.assert_array_equal(v[:,[0,2,3,5]],a["dense"][:,[0,2,3,5]])
            assert np.all(v>=0) and np.all(v[:,1]<=v[:,0]) and np.all(v[:,4]<=v[:,3])
            pair=c[[1,4]]/c[[0,3]]
            np.testing.assert_allclose(pair,metrics[k],rtol=1e-9)
            comparison[k]=float(pair.mean())
        report[stem]=dict(programs=len(v),original_queries=int(c[0]),current_queries=int(c[3]),balanced=comparison)
    print(json.dumps(report,indent=2))
if __name__=="__main__":main(Path(sys.argv[1]))
