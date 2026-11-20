"""Verify the compact-model validation threshold on the original recovered panel."""
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from verify import DATA, contrast, expected_panel


def main():
    root = DATA/"large-sealed-width-mid"
    report = json.loads((root/"result.json").read_text())
    receipt = json.loads((root/"receipt.json").read_text())
    assert receipt["status"]=="done" and receipt["revision"]=="6da2ada851d959e4fc252c5bda60f49882732b20"
    assert report["updates"]==4096 and report["parameters_per_compact_arm"]==238269312
    assert np.isclose(report["peak_rate"],.0004/3)
    w,g = expected_panel()
    groups = np.array(g)
    data = {}
    for k,f in (("sealed",root/"sealed-validation.npz"),("working",root/"working-validation.npz"),
                ("readable",DATA/"large-width-rate-mid/fixed-validation.npz")):
        with np.load(f) as z:
            np.testing.assert_array_equal(z["windows"],w)
            np.testing.assert_array_equal(z["groups"],groups)
            data[k] = z["scores"][:,0]
        assert np.isfinite(data[k]).all() and (data[k]>=0).all()
        if k in report["bpb"]:
            np.testing.assert_allclose(data[k].mean()/512,report["bpb"][k])
    checks = {}
    for k in ("readable","working"):
        name = "sealed_minus_"+k
        c = contrast(data["sealed"],data[k],groups)
        np.testing.assert_allclose([c["delta"],*c["interval"]],[report[name]["delta"],*report[name]["interval"]])
        assert c["packages"]==report[name]["packages"]
        checks[name] = c
    size_ok = checks["sealed_minus_readable"]["interval"][1]<=.02
    signal_ok = checks["sealed_minus_working"]["delta"]<=-.02 and checks["sealed_minus_working"]["interval"][1]<0
    assert report["decision"]==("advance" if size_ok or signal_ok else "stop")
    print(json.dumps(dict(bpb={k:float(v.mean()/512) for k,v in data.items()}, contrasts=checks,
                          qualifying_size=bool(size_ok), qualifying_early_copy=bool(signal_ok)),indent=2))


if __name__=="__main__":
    main()
