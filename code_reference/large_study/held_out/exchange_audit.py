"""Reconstruct paired identifier losses and exact partner answers at both widths."""
import json
from pathlib import Path
import random
import string
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "results"


def check(archive):
    root = DATA / archive
    report = json.loads((root / "result.json").read_text())
    receipt = json.loads((root / "receipt.json").read_text())
    assert receipt["status"] == "done" and receipt["exit"] == 0
    if archive == "large-wide-exchange":
        assert report["width"] == 1536 and report["lane"] == 192
        assert receipt["revision"] == "671f31066e9c0588869a8dda102374bb7cc6d7c7"
        assert report["checkpoint"] == "large-python-width-rate-main"
    targets = []
    for i in range(1024):
        rng = random.Random(f"cpython-ref-swap-20261119-{i}")
        rng.choices(string.ascii_lowercase,k=5)
        targets.append(list(map(ord,rng.sample(string.ascii_lowercase,2))))
    targets = np.array(targets)
    result = {}
    for arm in ("fixed","writable"):
        reference = report["models"][arm]
        with np.load(root/(arm+"-exchange.npz")) as z:
            data = dict(z)
        good = {}
        for k,value in data.items():
            assert value.shape == (1024,2,3) and np.isfinite(value).all()
            np.testing.assert_array_equal(value[:,:,0],targets)
            good[k] = (value[:,:,2]>0).all(1)
        np.testing.assert_array_equal(data["whole"][:,:,1],data["clean"][:,::-1,1])
        np.testing.assert_array_equal(data["whole"][:,:,2],-data["clean"][:,::-1,2])
        outcomes = {}
        for k in data:
            lost = good["clean"] & ~good[k]
            gained = good[k] & ~good["clean"]
            outcomes[k] = dict(solved=int(good[k].sum()), lost=int(lost.sum()), gained=int(gained.sum()),
                               both_switched=int((good["clean"] & (data[k][:,:,2]<0).all(1)).sum()))
            for target, name in (("solved","pairs"),("lost","lost"),("gained","gained")):
                assert outcomes[k][target] == reference["modes"][k][name]
        delta = (good["clean"] & ~good["designated"]).astype(int) - (good["clean"] & ~good["neighbor"])
        ix = np.random.default_rng(7851).choice(len(delta), (4000,len(delta)))
        ci = np.quantile(delta[ix].mean(1),[.025,.975]).tolist()
        assert reference["extra_designated_losses"] == int(delta.sum())
        np.testing.assert_allclose(ci,reference["interval"])
        outcome = good["clean"].mean()>.50 and delta.sum()>=32 and ci[0]>0
        assert bool(outcome) == reference["qualified"]
        result[arm] = dict(outcomes=outcomes, excess_fixed_losses=int(delta.sum()), interval=ci)
    return result


if __name__ == "__main__":
    print(json.dumps(dict(small=check("exchange"),large=check("large-wide-exchange")),indent=2))
