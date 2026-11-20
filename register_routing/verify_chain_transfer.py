"""Independently interpret the saved chains and audit paired model comparisons."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import sys
import numpy as np


def interpret(a, al, b, bl, c, payload, query, mode):
    def histories(keys, values):
        table=defaultdict(list)
        for i,key in enumerate(keys):
            table[tuple(key)].append((i,values[i]))
        return table
    first=histories(a,al);second=histories(b,bl);final=histories(c,payload)
    addresses=[]
    answers=[]
    for q,op in zip(query,mode):
        alias=first[tuple(q)][-1][1]
        key=second[tuple(alias)][-1][1]
        location,value=final[tuple(key)][-1 if op else 0]
        addresses.append(location)
        answers.append(value)
    return addresses,answers,first[tuple(query[0])]


def interval(x):
    rng=np.random.default_rng(771921)
    draws=np.concatenate([x[rng.integers(len(x),size=(200,len(x)))].mean(1) for _ in range(20)])
    return np.quantile(draws,[.025,.975])


def audit(folder, archive):
    report=json.loads((folder/"result.json").read_text())
    assert report["donor_archive_sha256"]==hashlib.sha256(archive.read_bytes()).hexdigest()
    audited={}
    gate=True
    for label,outcome in report["fits"].items():
        with np.load(folder/(label+".npz")) as a:
            fields=("a_keys","a_links","b_keys","b_links","c_keys","payloads","query","kind")
            tables=[a["input_"+k] for k in fields]
            addresses,expected,current,stale=[],[],[],[]
            for p,edited,control in zip(zip(*tables),a["edited_link"],a["stale_link"]):
                loc,answer,link=interpret(*p)
                assert len(link)==2 and len(set(p[5]))==60
                for changed,row in ((edited,link[-1][0]),(control,link[0][0])):
                    changed_rows=np.flatnonzero(np.any(p[1]!=changed,axis=1))
                    np.testing.assert_array_equal(changed_rows,[row])
                assert np.array_equal(edited[link[-1][0]],control[link[0][0]])
                def variant(new):
                    return interpret(p[0],new,*p[2:6],p[6][:2],p[7][:2])[1]
                current.append(variant(edited))
                stale.append(variant(control))
                assert np.all(np.array(current[-1])!=answer[:2])
                assert stale[-1]==answer[:2]
                addresses.append(loc)
                expected.append(answer)
            for key,truth in (("address",addresses),("gold",expected),("current_gold",current),("stale_gold",stale)):
                np.testing.assert_array_equal(a[key],truth)
            expected=np.array(expected)
            assert expected.shape==(1024,16)
            observed=outcome["held_out"]
            hits={}
            for arm,r in observed["arms"].items():
                hit=a[arm+"_answer"]==expected
                hits[arm]=hit
                np.testing.assert_allclose([hit.mean(),(a[arm+"_address"]==addresses).mean()],
                                           [r["accuracy"],r["address_accuracy"]])
            clean=hits["graft"][:,:2]
            local_gate=True
            for k in ("scratch","pretrained","rolled"):
                diff=hits["graft"].mean(1)-hits[k].mean(1)
                contrast=observed["contrasts"][k]
                np.testing.assert_allclose(diff.mean(),contrast["delta"])
                np.testing.assert_allclose(interval(diff),contrast["interval"])
                local_gate&=bool(contrast["interval"][0]>0 and (k=="rolled" or diff.mean()>=.03))
            for k in ("current","stale"):
                correct=a[k+"_answer"]==a[k+"_gold"]
                c=observed["causal"][k]
                assert c==dict(clean_correct=int(clean.sum()),both_correct=int((correct&clean).sum()),
                    correct_unconditional=int(correct.sum()),all_queries=correct.size)
                local_gate&=bool(clean.sum() and c["both_correct"]>=.9*clean.sum())
            assert outcome["passes"]==local_gate
            gate&=local_gate
            audited[label]=dict(programs=len(expected),queries=expected.size,clean_graft=int(hits["graft"].sum()),
                clean_with_own_pretraining=int(hits["pretrained"].sum()),verified=True)
    assert gate==report["advantage_replicates"]
    return audited


if __name__=="__main__":
    print(json.dumps(audit(Path(sys.argv[1]), Path(sys.argv[2])),indent=2))
