"""Reconstruct the stable-minus-random contrast from matched program counts."""
import json
from pathlib import Path
import sys
import numpy as np

def balanced(counts):
    return (counts[...,1]/counts[...,0]+counts[...,4]/counts[...,3])/2

def main(folder,original):
    fitted=json.loads((folder/"result.json").read_text());audit={}
    for name in ("validation","test"):
        panel=np.load(folder/(name+".npz"));prior=np.load(original/(name+".npz"))
        for arm in ("matched","rotated","dense"):np.testing.assert_array_equal(panel[arm],prior[arm])
        values={}
        for arm in panel.files:
            c=panel[arm]
            np.testing.assert_array_equal(c[:,[0,2,3,5]],panel["dense"][:,[0,2,3,5]])
            assert np.all(c[:,1]<=c[:,0]) and np.all(c[:,4]<=c[:,3])
            total=c.sum(0)
            rates=total[[1,4]]/total[[0,3]]
            np.testing.assert_allclose(rates,fitted["metrics"][name]["arms"][arm])
            values[arm]=float(rates.mean())
        x,y=panel["rotated"],panel["random"]
        diff=balanced(x.sum(0))-balanced(y.sum(0))
        samples=[];rng=np.random.default_rng(192841)
        for _ in range(80):
            pick=rng.choice(len(y),size=(50,len(y)),replace=True)
            samples.extend(balanced(x[pick].sum(1))-balanced(y[pick].sum(1)))
        ci=np.quantile(samples,[.025,.975]).tolist()
        np.testing.assert_allclose(ci,fitted["metrics"][name]["interval"])
        np.testing.assert_allclose(diff,fitted["metrics"][name]["stable_minus_random"])
        audit[name]=dict(balanced=values,queries=x.sum(0)[[0,3]].tolist(),programs=len(x),stable_minus_random=diff,interval=ci)
    print(json.dumps(audit,indent=2))
if __name__=="__main__":main(*map(Path,sys.argv[1:]))
