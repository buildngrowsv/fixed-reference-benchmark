"""Verify the released location-transfer evidence without MLX."""
from pathlib import Path
import json
import sys
import numpy as np

def bootstrap(d):
    rng=np.random.default_rng(771921)
    draws=np.concatenate([d[rng.integers(len(d),size=(200,len(d)))].mean(1) for _ in range(20)])
    return np.quantile(draws,[.025,.975])

def audit(folder):
    s=json.loads((folder/"result.json").read_text())
    report={}
    criteria=[]
    for seed, trial in s["seeds"].items():
        with np.load(folder/f"paired-{seed}.npz") as data:
            k,q,v=data["keys"],data["queries"],data["values"]
            assert k.shape==(1024,60,2) and q.shape==(1024,16,2)
            assert np.all(k.sum(-1)%2==1) and np.all(q.sum(-1)%2==1)
            assert np.all(q[:,::2]==q[:,1::2])
            marks=[]
            for keys,queries,values in zip(k,q,v):
                assert len(set(values))==60 and 64<=values.min()<=values.max()<128
                occurrences={}
                for j,key in enumerate(keys):
                    occurrences.setdefault(tuple(key),[]).append(j)
                assert len(occurrences)==20 and all(len(a)==3 for a in occurrences.values())
                marks.append([occurrences[tuple(key)][-1 if j%2 else 0] for j,key in enumerate(queries)])
            src=np.array(marks)
            np.testing.assert_array_equal(src,data["address"])
            gold=np.take_along_axis(v,src,axis=1)
            metrics=trial["held_out"]
            all_ok={}
            for arm,result in metrics["arms"].items():
                hit=data[f"{arm}_answers"]==gold-(64 if arm=="donor" else 0)
                all_ok[arm]=hit
                np.testing.assert_allclose([hit.mean(),hit[:,::2].mean(),hit[:,1::2].mean(),
                    (data[f"{arm}_p"]==src).mean()],[result[x] for x in ("accuracy","first","latest","source")])
            clean=all_ok["graft"]
            clean_both=clean&(data["graft_p"]==src)
            for k_name,effect in metrics["contrast"].items():
                delta=clean.mean(1)-all_ok[k_name].mean(1)
                np.testing.assert_allclose(delta.mean(),effect["delta"])
                np.testing.assert_allclose(bootstrap(delta),effect["interval"])
            assert np.all(data["gold_changed"] != v)
            target=data["replace_actual"]==data["gold_changed"]
            stale=data["stale_actual"]==gold
            np.testing.assert_array_equal(target,data["replace_correct"])
            np.testing.assert_array_equal(stale,data["stale_correct"])
            time_correct=clean_both.reshape(-1,8,2).all(2)
            np.testing.assert_array_equal(time_correct,data["time_ok"])
            key_good=np.zeros((len(q),2),dtype=bool)
            key_exists=key_good.copy()
            for j,queries in enumerate(q[:,::2]):
                for half in (0,1):
                    pairs=[(a,b) for a in range(8) for b in range(a+1,8)
                        if queries[a,half]==queries[b,half] and queries[a,1-half]!=queries[b,1-half]]
                    if pairs:
                        a,b=pairs[0]
                        key_exists[j,half]=True
                        key_good[j,half]=clean_both[j,2*a]&clean_both[j,2*b]
            np.testing.assert_array_equal(key_exists,data["key_have"])
            np.testing.assert_array_equal(key_good&key_exists,data["key_ok"]&data["key_have"])
            counter=dict(key_switched=int(key_good.sum()),key_pairs=int(key_exists.sum()),
                time_switched=int(time_correct.sum()),time_pairs=time_correct.size,clean_correct=int(clean.sum()),
                queries=clean.size,target_replaced=int((target&clean).sum()),stale_preserved=int((stale&clean).sum()),
                target_unconditional=int(target.sum()),stale_unconditional=int(stale.sum()))
            assert counter==metrics["counterfactual"]
            boot=metrics["contrast"]
            passes=(metrics["arms"]["graft"]["source"]>=.9 and clean.mean()>=.9 and
                boot["permuted"]["interval"][0]>.10 and counter["target_replaced"]>=.9*clean.sum() and
                counter["stale_preserved"]>=.9*clean.sum() and key_good.sum()>=.8*key_exists.sum() and
                time_correct.sum()>=.8*time_correct.size)
            criteria.append((passes,boot["native"]["interval"][0]>.05))
            report[seed]=dict(programs=len(q),queries=clean.size,graft_answers=int(clean.sum()),
                hard_read_at_graft_argmax=int((data["graft_p"]==src).sum()),target_replaced=counter["target_replaced"],
                stale_preserved=counter["stale_preserved"],recomputed=True)
    assert s["location_transfer"]==all(p for p,_ in criteria)
    assert s["native_advantage"]==all(p and a for p,a in criteria)
    return report

if __name__=="__main__":
    print(json.dumps(audit(Path(sys.argv[1])),indent=2))
