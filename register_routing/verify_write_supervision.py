"""Independently verify the unconditional answer accuracy and clean loss gate."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

from verify_chain_transfer import interval
from verify_paired_edits import audit_labels

ARMS=("donor","scratch","pretrained")
REGIMES=("clean","answer_edits","write")
PHASES=("clean","current","stale")


def verify(folder,replay):
    report=json.loads((folder/"result.json").read_text())
    audit={}
    for label,run in report["fits"].items():
        sha=hashlib.sha256((replay/(label+".npz")).read_bytes()).hexdigest()
        assert sha==run["previous_panel_sha256"]
        trial=run["primary"]
        with np.load(folder/(label+".npz")) as data:
            payload=data["input_payloads"]
            assert payload.shape==(2048,60)
            assert (np.diff(np.sort(payload,axis=1),axis=1)>0).all()
            audit_labels(data)
            scores={}
            for arm in ARMS:
                for regime in REGIMES:
                    k=arm+"_"+regime
                    expected=trial["scores"][k]
                    hits={}
                    for p in PHASES:
                        pos=data[k+"_"+p+"_source"]
                        # A different indexing formulation from the experiment's gather.
                        selected=payload[np.arange(len(pos))[:,None],pos]
                        hit=selected==data[p+"_gold"]
                        hits[p]=hit
                        row=expected[p]
                        assert row["n"]==hit.size and row["correct"]==hit.sum()
                        np.testing.assert_allclose(row["accuracy"],hit.mean())
                        np.testing.assert_allclose(row["interval"],interval(hit.mean(1)))
                    scores[k]={k:v.mean(1) for k,v in hits.items()}
                    both=hits["current"]&hits["stale"]
                    np.testing.assert_allclose(expected["both_edits"],both.mean())
                    np.testing.assert_allclose(expected["complete_edited_program"],both.all(1).mean())
                    np.testing.assert_allclose(expected["clean_neural_readout"],
                        (data[k+"_clean_soft"]==data["clean_gold"]).mean())
            verdict={}
            for arm in ARMS:
                for r in REGIMES[:2]:
                    for phase in PHASES:
                        compared=trial["contrasts"][arm+"_vs_"+r][phase]
                        d=scores[arm+"_write"][phase]-scores[arm+"_"+r][phase]
                        np.testing.assert_allclose(compared["delta"],d.mean())
                        np.testing.assert_allclose(compared["interval"],interval(d))
                d=trial["contrasts"][arm+"_vs_clean"]["clean"]
                e=trial["scores"][arm+"_write"]
                verdict[arm]=dict(clears_both_edits=min(e["current"]["accuracy"],e["stale"]["accuracy"])>=.9,
                    both_lower_bounds_above_90=min(e["current"]["interval"][0],e["stale"]["interval"][0])>.9,
                    clean_held=d["delta"]>=0 and d["interval"][0]>-report["clean_loss_margin"])
            assert verdict==trial["decision"]
            audit[label]=dict(programs=len(payload),all_edit_queries_per_type=4096,
                donor_correct={p:trial["scores"]["donor_write"][p]["correct"] for p in PHASES},
                verified=verdict)
    flags=[o["verified"]["donor"] for o in audit.values()]
    assert report["passes"]==all(f["clears_both_edits"] and f["clean_held"] for f in flags)
    assert report["strict_edit_intervals"]==all(f["both_lower_bounds_above_90"] for f in flags)
    print(json.dumps(audit,indent=2))


if __name__=="__main__":
    verify(*(Path(p) for p in sys.argv[1:]))
