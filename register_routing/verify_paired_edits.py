"""Reconstruct held-out labels and all reported paired edit rates without MLX."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

from verify_chain_transfer import interpret, interval

ARMS=("donor","scratch","pretrained")
REGIMES=("clean","paired")


def audit_labels(a):
    fields=("a_keys","a_links","b_keys","b_links","c_keys","payloads","query","kind")
    clean,active,obsolete=[],[],[]
    for p,change,sham in zip(zip(*(a["input_"+f] for f in fields)),a["current_link"],a["stale_link"]):
        loc,val,link=interpret(*p)
        assert len(link)==2
        clean.append((loc,val))
        for table,row,collection in ((change,link[-1][0],active),(sham,link[0][0],obsolete)):
            np.testing.assert_array_equal(np.flatnonzero(np.any(table!=p[1],axis=1)),[row])
            l,v,_=interpret(p[0],table,*p[2:6],p[6][:2],p[7][:2])
            collection.append((l,v))
        assert np.array_equal(change[link[-1][0]],sham[link[0][0]])
        assert np.all(np.array(active[-1][1])!=val[:2]) and obsolete[-1][1]==val[:2]
    for name,rows in (("clean",clean),("current",active),("stale",obsolete)):
        pos,answer=zip(*rows)
        np.testing.assert_array_equal(a[name+"_source"],pos)
        np.testing.assert_array_equal(a[name+"_gold"],answer)


def audit(root, previous, adapted):
    r=json.loads((root/"result.json").read_text())
    results={}
    for name,trial in r["fits"].items():
        for k,base in (("chain_study",previous),("adaptation",adapted)):
            assert hashlib.sha256((base/(name+".npz")).read_bytes()).hexdigest()==trial["archived_hashes"][k]
        h=trial["held_out"]
        with np.load(root/(name+".npz")) as a:
            audit_labels(a)
            draws=np.random.default_rng(771921).integers(len(a["clean_gold"]),size=(4000,len(a["clean_gold"])))
            rates={}
            programs={}
            for arm in ARMS:
                for regime in REGIMES:
                    k=arm+"_"+regime
                    row=h["scores"][k]
                    clean=a[k+"_clean"]==a["clean_gold"]
                    assert clean.shape==(1024,16)
                    programs[k]=clean.mean(1)
                    np.testing.assert_allclose(row["clean"],clean.mean())
                    for e,d in row["edits"].items():
                        ok=a[k+"_"+e]==a[e+"_gold"]
                        eligible=clean[:,:2]
                        n=eligible.sum()
                        hits=(eligible&ok).sum()
                        assert (int(n),int(hits),int(ok.sum()))==(d["clean_correct"],d["both_correct"],d["edit_correct"])
                        np.testing.assert_allclose([hits/n,ok.mean()],[d["conditional"],d["unconditional"]])
                        picked=eligible.sum(1)[draws].sum(1)
                        matched=(eligible&ok).sum(1)[draws].sum(1)
                        np.testing.assert_allclose(d["interval"],np.quantile(matched/picked,[.025,.975]))
            for k,d in h["changes"].items():
                delta=programs[k+"_paired"]-programs[k+"_clean"]
                np.testing.assert_allclose(d["delta"],delta.mean())
                np.testing.assert_allclose(d["interval"],interval(delta))
            for k,d in h["donor_vs_paired_native"].items():
                delta=programs["donor_paired"]-programs[k+"_paired"]
                np.testing.assert_allclose(d["delta"],delta.mean())
                np.testing.assert_allclose(d["interval"],interval(delta))
            summary=h["scores"]["donor_paired"]
            assert h["qualifies"]==(h["changes"]["donor"]["interval"][0]>-r["clean_margin"]
                and all(e["conditional"]>=.9 for e in summary["edits"].values()))
            results[name]=dict(programs=len(clean),checked_counterfactuals=2*len(clean),
                donor_clean=summary["clean"],donor_edits={k:v["conditional"] for k,v in summary["edits"].items()},
                qualified=h["qualifies"])
    assert r["qualifies"]==all(f["qualified"] for f in results.values())
    print(json.dumps(results,indent=2))


if __name__=="__main__":
    audit(*(Path(p) for p in sys.argv[1:]))
