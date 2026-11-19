"""Aggregate paired counts. The two members of a corruption pair are resampled together."""
import json
from pathlib import Path
import numpy as np

R=Path(__file__).resolve().parent/"results"
SEEDS=(101,211,307)


def effect(a,b):
    assert np.array_equal(a[:,:,0],b[:,:,0])
    d=(a[:,:,1]-b[:,:,1]).reshape(3,64,2).sum(-1)
    n=a[:,:,0].reshape(3,64,2).sum(-1)
    rng=np.random.default_rng(47173)
    sims=[]
    for _ in range(20):
        i=rng.integers(64,size=(250,64))
        j=rng.integers(3,size=(250,3))
        pick=(j[:,:,None],i[:,None,:])
        sims.extend((d[pick].sum((1,2))/n[pick].sum((1,2))).tolist())
    return [float(d.sum()/n.sum()),*np.quantile(sims,[.025,.975]).tolist()]


def main():
    arrays={}; summary={}
    for mode in ("assigned","neutral","exchange"):
        grouped={}; chosen=[]; seconds=[]; mem=[]; curves=[]
        for seed in SEEDS:
            rec=json.loads((R/f"{mode}-{seed}.json").read_text());j=rec["address"]
            assert j==int(np.argmax(rec["validation"][-1]["probe"]))
            chosen.append(j);seconds.append(rec["train_seconds"]);mem.append(rec["peak_bytes"])
            curves.append([row["exact"] for row in rec["validation"]])
            with np.load(R/f"{mode}-{seed}.npz") as d:
                mapping={"primary":"primary","baseline":"from_memory","selected_swap":f"corrupt{j}",
                         "other_swap":f"corrupt{1-j}","selected_probe":f"probe{j}","other_probe":f"probe{1-j}"}
                for key,source in mapping.items(): grouped.setdefault(key,[]).append(d[source])
        arr={k:np.stack(v) for k,v in grouped.items()}
        arrays[mode]=arr
        metrics={}
        for k,v in arr.items():
            assert v.shape==(3,128,3)
            n,h,b=v.sum((0,1));assert n>0 and 32*h<=b<=32*n
            metrics[k]=dict(sites=int(v[0,:,0].sum()),exact=h/n,bit=b/(32*n),
                            seed_exact=(v[:,:,1].sum(1)/v[:,:,0].sum(1)).tolist())
        summary[mode]=dict(metrics=metrics,selectivity=effect(arr["other_swap"],arr["selected_swap"]),
                           addresses=chosen,params=rec["params"],mean_curve=np.mean(curves,axis=0).tolist(),
                           training_seconds=seconds,peak_bytes=mem)
    comparisons={k+"_minus_assigned":effect(arrays[k]["primary"],arrays["assigned"]["primary"])
                 for k in ("neutral","exchange")}
    comparisons["neutral_minus_exchange"]=effect(arrays["neutral"]["primary"],arrays["exchange"]["primary"])
    (R/"summary.json").write_text(json.dumps(dict(conditions=summary,comparisons=comparisons),indent=2)+"\n")
    for name,s in summary.items():
        print(name,{k:round(v["exact"]*100,3) for k,v in s["metrics"].items()},s["selectivity"])
    print(comparisons)


if __name__=="__main__": main()
