"""Paired, color-stratified bootstrap over the shared test boards and training seeds."""
import csv
import json
from pathlib import Path
import numpy as np


def interval(diff, strata):
    rng=np.random.default_rng(92181)
    draws=10000
    seeds, boards=diff.shape
    sums=np.zeros((draws,seeds))
    for z in np.unique(strata):
        subset=diff[:,strata==z]
        patterns,counts=np.unique(subset.T,axis=0,return_counts=True)
        sampled=rng.multinomial(subset.shape[1],counts/counts.sum(),size=draws)
        sums+=sampled@patterns
    sampled_seeds=rng.integers(seeds,size=(draws,seeds))
    effects=np.take_along_axis(sums/boards,sampled_seeds,axis=1).mean(axis=1)
    return np.quantile(effects,[.025,.975]).tolist()


def summarize(root,area,seeds,variants):
    balanced=root.name=="balanced"
    with np.load(root/"data"/"sensitive.npz") as raw:
        final,start=raw["final"],raw["start"]
        assert raw["sensitive"].all()
    with np.load(root/"data"/"test.npz") as raw:
        natural_current,natural_initial=raw["final"],raw["start"]
    primary,natural={},{}
    rows=[]
    for variant in variants:
        p,n=[],[]
        for seed in seeds:
            d=area/f"{seed}-{variant}"
            r=json.loads((d/"result.json").read_text())
            with np.load(d/"predictions.npz") as f:
                pred=f["sensitive"][:,1]
                assert abs((pred==final).mean()-r["sensitive"]["current"])<1e-9
                p.append(pred==final)
                n.append(f["test"][:,1]==natural_current)
            row=dict(variant=variant,seed=seed,original=r["test"]["original"],current=r["test"]["current"],
                     restore=r["sensitive"]["current"],restore_changed=float((pred[final!=start]==final[final!=start]).mean()),
                     restore_unchanged=float((pred[final==start]==final[final==start]).mean()),
                     current_log_loss=r["test"]["log_loss"][1],parameters=r["parameters"],
                     seconds=r["training_seconds"],peak_mib=r["train_peak_mib"],epochs=r["epochs"])
            if variant=="fixed":
                row.update(r["sensitive"]["patch"])
            rows.append(row)
        primary[variant]=np.stack(p)
        natural[variant]=np.stack(n)
    with (area/"per-seed.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)
    summary={}
    for v in variants:
        these=[row for row in rows if row["variant"]==v]
        summary[v]={k:float(np.mean([r[k] for r in these])) for k in these[0] if k not in ("variant","seed")}
    pair={}
    strata=final+4*(final!=start) if balanced else final
    for v in variants[:-1]:
        delta=primary["fixed"].astype("int8")-primary[v].astype("int8")
        other=natural["fixed"].astype("int8")-natural[v].astype("int8")
        pair[v]=dict(mean=float(delta.mean()),seeds=delta.mean(axis=1).tolist(),
                     wins=int((delta.mean(axis=1)>0).sum()),interval=interval(delta,strata),
                     natural=float(other.mean()))
    baseline=dict(restore=float((final==start).mean()),current=float((natural_current==natural_initial).mean()))
    out=dict(summary=summary,fixed_differences=pair,copy_start_baseline=baseline,
             balanced=balanced,training_seeds=seeds,bootstrap_draws=10000,boards=len(final))
    (area/"summary.json").write_text(json.dumps(out,indent=2)+"\n")
    print("variant       restore   current   original   changed   seconds")
    for v,s in summary.items():
        print(f"{v:10} {s['restore']:9.4f} {s['current']:9.4f} {s['original']:9.4f} {s['restore_changed']:9.4f} {s['seconds']:7.2f}")
    for v,g in pair.items():
        print("fixed minus",v,round(g["mean"]*100,3),[round(x*100,3) for x in g["interval"]],"wins",g["wins"])
    print("copy initial",baseline,flush=True)
