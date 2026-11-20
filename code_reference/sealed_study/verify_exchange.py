"""Audit the identifier-pair outcome at the defining token."""
from pathlib import Path
import json
import numpy as np
folder=Path(__file__).resolve().parents[1]/"results/sealed-exchange"

def audit():
    summary=json.loads((folder/"result.json").read_text())
    for seed,r in summary["seeds"].items():
        z=np.load(folder/("seed-"+seed+".npz"));clean=z["clean"]
        assert clean.shape==(1024,2,3)
        assert np.all(clean[:,0,0]!=clean[:,1,0])
        for a in z.values():
            np.testing.assert_array_equal(a[:,:,0],clean[:,:,0])
        np.testing.assert_array_equal(z["whole"][:,:,2],-clean[:,::-1,2])
        np.testing.assert_array_equal(z["whole"][:,:,1],clean[:,::-1,1])
        good=(clean[:,:,2]>0).all(1); top=(clean[:,:,1]==clean[:,:,0]).all(1)
        assert (good.sum(),top.sum())==(r["clean_pairs"],r["clean_top_one_pairs"])
        flips={k:good & (z[k][:,:,2]<0).all(1) for k in z.files}
        for k,a in z.items():
            assert int(flips[k].sum())==r["double_reversals"][k]
            assert (top&(a[:,:,1]==clean[:,::-1,0]).all(1)).sum()==r["exact_partner_answers"][k]
        diff=flips["reference"].astype(float)-flips["working"]
        rng=np.random.default_rng(393217)
        draws=np.concatenate([diff[rng.choice(len(diff),(50,len(diff)))].mean(1) for _ in range(80)])
        np.testing.assert_allclose([diff.mean(),*np.quantile(draws,[.025,.975])],[r["difference"],*r["interval"]])
    return summary["seeds"]

if __name__=="__main__":
    print(json.dumps(audit(),indent=2))
