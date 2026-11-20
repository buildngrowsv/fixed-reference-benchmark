"""Recover the wide-model stage decision and its package-paired intervals."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "results"


def expected_panel(split="val", seed=881731, count=1024):
    records = json.loads((DATA/"main/corpus.json").read_text())["files"][split]
    lengths = np.array([r["bytes"]-512 for r in records])
    ends = lengths.cumsum()
    starts = ends-lengths
    windows, groups = [], []
    for start in range(0,count,16):
        locations = np.random.default_rng([seed,start]).integers(ends[-1],size=16)
        indices = np.searchsorted(ends, locations, side="right")
        windows.extend(zip(indices,locations-starts[indices]))
        groups.extend(records[i]["group"] for i in indices)
    return windows, groups


def contrast(first, second, groups):
    packages = np.unique(groups)
    delta = (first-second)/512
    total = np.array([delta[groups==p].sum() for p in packages])
    count = np.array([(groups==p).sum() for p in packages])
    samples = np.random.default_rng(9651).integers(len(packages), size=(5000,len(packages)))
    estimates = total[samples].sum(1)/count[samples].sum(1)
    return dict(delta=float(delta.mean()), interval=np.quantile(estimates,[.025,.975]).tolist(),packages=len(packages))


def audit():
    folder = DATA/"large-original-mid"
    report = json.loads((folder/"result.json").read_text())
    small = json.loads((DATA/"mid/result.json").read_text())
    windows, groups = expected_panel()
    groups = np.array(groups)
    arrays = {}
    performance = {}
    def read_scores(path):
        with np.load(path) as f:
            np.testing.assert_array_equal(f["windows"],windows)
            np.testing.assert_array_equal(f["groups"],groups)
            values = f["scores"]
            assert len(values)==1024 and np.all(np.isfinite(values)) and np.all(values>=0)
            return values
    for size, root in (("small",DATA/"mid"),("large",folder)):
        for arm in ("fixed","writable"):
            name = size+"_"+arm
            v = read_scores(root/(arm+"-validation.npz"))
            assert np.all(v[:,1]<=512)
            arrays[name] = v[:,0]
            performance[name] = dict(bpb=float(v[:,0].mean()/512), next_byte_accuracy=float(v[:,1].mean()/512))
            source = small["models"] if size=="small" else report["arms"]
            np.testing.assert_allclose(performance[name]["bpb"],source[arm]["validation_bpb"])
    computed = {}
    for size in ("small","large"):
        fixed, writable = arrays[size+"_fixed"], arrays[size+"_writable"]
        computed[size] = contrast(fixed, writable, groups)
        arrays[size+"_gap"] = fixed-writable
    computed["size_interaction"] = contrast(arrays["large_gap"],arrays["small_gap"],groups)
    for name, c in computed.items():
        record = report["contrasts"]["validation"][name]
        np.testing.assert_allclose([c["delta"],*c["interval"]],[record["delta"],*record["interval"]])
        assert c["packages"]==record["packages"]
    controls = read_scores(folder/"validation-controls.npz")
    for k,col in (("unigram",0),("bigram",2)):
        performance[k] = dict(bpb=float(controls[:,col].mean()/512),next_byte_accuracy=float(controls[:,col+1].mean()/512))
    np.testing.assert_allclose(performance["unigram"]["bpb"], small["validation_control"]["unigram_bpb"])
    ok = max(performance["large_"+k]["bpb"] for k in ("fixed","writable"))<3.24
    assert report["decision"]==("advance" if ok else "stop")
    return dict(validation_windows=len(groups),validation_packages=len(set(groups)), performance=performance,
                fixed_minus_writable=computed,passed_competence_gate=ok)


if __name__=="__main__":
    print(json.dumps(audit(),indent=2))
