"""Recompute fresh discrete-hop outcomes from positions and independently interpreted tables."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from verify_chain_transfer import interval
from verify_paired_edits import audit_labels

PHASES=("clean","current","stale")


def verify(folder, replay):
    report=json.loads((folder/"result.json").read_text())
    audited={}
    for name,run in report["fits"].items():
        assert hashlib.sha256((replay/(name+".npz")).read_bytes()).hexdigest()==run["archive_sha256"]
        trial=run["primary"]
        with np.load(folder/(name+".npz")) as a:
            audit_labels(a)
            assert a["input_payloads"].shape==(2048,60)
            hits={}
            for k,record in trial["scores"].items():
                init,fit,path=k.split("_")
                results={}
                for p in PHASES:
                    inds=a[f"{init}_{fit}_{p}_{path}"]
                    gold=a[p+"_gold"]
                    assert inds.shape==gold.shape and inds.max()<60
                    v=a["input_payloads"][np.arange(len(inds))[:,None],inds]
                    result=v==gold
                    results[p]=result
                    stats=record[p]
                    assert (stats["correct"],stats["queries"])==(result.sum(),result.size)
                    np.testing.assert_allclose(stats["accuracy"],result.mean())
                    np.testing.assert_allclose(stats["interval"],interval(result.mean(1)))
                hits[k]=results
                paired=results["current"]&results["stale"]
                np.testing.assert_allclose([record["both_edits"],record["complete_edited_program"]],
                                           [paired.mean(),paired.all(1).mean()])
            for k,phases in trial["changes"].items():
                init=k.split("_",1)[0]
                for p,d in phases.items():
                    diff=hits[k][p].mean(1)-hits[init+"_clean_soft"][p].mean(1)
                    np.testing.assert_allclose(d["delta"],diff.mean())
                    np.testing.assert_allclose(d["interval"],interval(diff))
                e=trial["scores"][k]
                clean=phases["clean"]
                flags=dict(clears_edits=min(e[p]["accuracy"] for p in PHASES[1:])>=.9,
                    edit_intervals_clear=min(e[p]["interval"][0] for p in PHASES[1:])>.9,
                    clean_held=clean["delta"]>=0 and clean["interval"][0]>-.01)
                assert flags==trial["decision"][k]
            audited[name]=dict(programs=2048,queries_per_edit_type=4096,
                decisions=trial["decision"])
    print(json.dumps(audited,indent=2))


if __name__=="__main__":
    verify(*(Path(v) for v in sys.argv[1:]))
