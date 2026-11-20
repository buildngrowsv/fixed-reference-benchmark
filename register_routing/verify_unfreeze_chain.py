"""Independently check the unfrozen comparison against the audited reference arrays."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

from verify_chain_transfer import interval


def main(folder, baseline):
    s=json.loads((folder/"result.json").read_text())
    audited={}
    for name,fit in s["fits"].items():
        archived=baseline/(name+".npz")
        assert hashlib.sha256(archived.read_bytes()).hexdigest()==fit["baseline_sha256"]
        with np.load(archived) as reference, np.load(folder/(name+".npz")) as new:
            gold=reference["gold"]
            assert gold.shape==(1024,16)
            candidates={"adapted":new["graft_answer"],"frozen":reference["graft_answer"],
                        "pretrained_native":reference["pretrained_answer"],"shifted":new["rolled_answer"]}
            hits={k:v==gold for k,v in candidates.items()}
            for k,hit in hits.items():
                np.testing.assert_allclose(hit.mean(),fit["accuracy"][k])
            for k,c in fit["contrasts"].items():
                d=hits["adapted"].mean(1)-hits[k].mean(1)
                np.testing.assert_allclose(d.mean(),c["delta"])
                np.testing.assert_allclose(interval(d),c["interval"])
            for k,prediction in (("adapted",new["graft_address"]),("frozen",reference["graft_address"]),
                                 ("pretrained_native",reference["pretrained_address"])):
                np.testing.assert_allclose((prediction==reference["address"]).mean(),fit["source_accuracy"][k])
            for arm,data in (("adapted",new),("frozen",reference)):
                clean=hits[arm][:,:2]
                for key in ("current","stale"):
                    ok=data[key+"_answer"]==reference[key+"_gold"]
                    assert fit["causal"][arm][key]==dict(clean_correct=int(clean.sum()),
                        both_correct=int((clean&ok).sum()),correct_unconditional=int(ok.sum()),all_queries=ok.size)
            rate=fit["accuracy"]
            np.testing.assert_allclose((rate["adapted"]-rate["frozen"])/(rate["pretrained_native"]-rate["frozen"]),
                                       fit["fraction_point_gap_recovered"])
            assert fit["passes_margin"]==(fit["contrasts"]["pretrained_native"]["interval"][0]>-s["margin"])
        a,f=(fit["training"][k] for k in ("adapted","frozen"))
        digest=s["identical_initial_router_sha256"]
        assert f["initial_router_sha256"]==f["final_router_sha256"]==a["initial_router_sha256"]==digest
        assert a["final_router_sha256"]!=digest and a["changed_router_weights"]>0 and f["changed_router_weights"]==0
        audited[name]=dict(programs=len(gold),adapted_correct=int(hits["adapted"].sum()),
                           frozen_correct=int(hits["frozen"].sum()),native_correct=int(hits["pretrained_native"].sum()),
                           checked=True)
    assert s["closes_gap"]==all(d["passes_margin"] for d in s["fits"].values())
    print(json.dumps(audited,indent=2))


if __name__=="__main__":
    main(Path(sys.argv[1]),Path(sys.argv[2]))
