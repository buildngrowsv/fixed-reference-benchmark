"""Full saved-weight GPU replay plus a raw-score cross-engine audit on eight cubes."""
import json
import numpy as np
import mlx.core as mx
from experiment import R, SEEDS, Assigned, Banks, panel, evaluate, rollout, save_json
from replay import inputs, predict


def main():
    rows=[]
    independent=inputs()
    for mode in ("assigned","neutral","exchange"):
        for seed in SEEDS:
            rec=json.loads((R/f"{mode}-{seed}.json").read_text())
            w={k:np.array(v,dtype=np.float32) for k,v in rec["weights"].items()}
            m=Assigned() if mode=="assigned" else Banks()
            m.load_weights([(k,mx.array(v)) for k,v in w.items()])
            data=list(panel(984731,6,16,16))
            rescored=evaluate(m,mode,data,True)
            with np.load(R/f"{mode}-{seed}.npz") as original:
                for k,v in rescored.items(): np.testing.assert_array_equal(v,original[k])
            x,a,s,_,_,flip=data[0]
            np.testing.assert_array_equal(x.reshape(independent[0].shape),independent[0])
            args=tuple(map(mx.array,(x,a,s,flip if mode=="exchange" else np.zeros_like(flip))))
            max_error=0.;largest_flip=0.;flip_count=0
            for corrupt in (None,0,1):
                device,midpoints=rollout(m,*args,corrupt=corrupt)
                cpu,mid=predict(w,independent,mode,corrupt)
                pairs=[(np.array(device).reshape(cpu.shape),cpu)]
                if corrupt is None: pairs+= [(np.array(d).reshape(c.shape),c) for d,c in zip(midpoints,mid)]
                for d,c in pairs:
                    error=np.abs(d-c)
                    max_error=max(max_error,float(error.max()))
                    flips=(d>0)!=(c>0)
                    flip_count+=int(flips.sum())
                    if flips.any(): largest_flip=max(largest_flip,float(np.maximum(np.abs(d[flips]),np.abs(c[flips])).max()))
            rows.append(dict(mode=mode,seed=seed,max_absolute_score_difference=max_error,
                             changed_signs=flip_count,max_magnitude_at_changed_sign=largest_flip))
    save_json(R/"device-audit.json",dict(exact_all_counts=True,full_panel_volumes=128,
                                          cross_engine_volumes=8,scores_compared_per_fit=5*8*216*32,rows=rows))
    print("All saved-weight GPU counts agree. Largest cross-engine difference, largest sign-flip magnitude:",
          max(r["max_absolute_score_difference"] for r in rows),max(r["max_magnitude_at_changed_sign"] for r in rows))


if __name__=="__main__": main()
