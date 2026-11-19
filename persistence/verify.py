import json
import numpy as np
from experiment_persistence import OUT,ARMS,load_or_fit,evaluate,panel
from replay import inputs,predict,count


def main():
    decision=json.loads((OUT/"decision.json").read_text())
    delta=decision["validation"]["persistent"]-decision["validation"]["neutral"]
    assert decision["qualified"]==(delta>=.05)
    control=inputs(752916); audit=[]
    for seed in decision["seeds"]:
        for arm in ARMS:
            model,r=load_or_fit(arm,seed)
            w={k:np.array(v,dtype="float32") for k,v in r["weights"].items()}
            full=evaluate(model,arm,panel(752916,6,16,16),True)
            with np.load(OUT/f"{arm}-{seed}.npz") as prior:
                for k,v in full.items(): np.testing.assert_array_equal(v,prior[k])
                z,half=predict(w,control,arm)
                _,_,_,_,target,mask,truth,dist,uses=control
                independent={"primary":count(z,target,mask),"from_memory":count(z,target,uses)}
                for j in range(2):
                    independent[f"probe{j}"]=count(half[j],truth,dist)
                    independent[f"corrupt{j}"]=count(predict(w,control,arm,j)[0],target,uses)
                for k,c in independent.items():
                    d=np.abs(c-prior[k][:8]).sum(0)
                    assert d[0]==0
                    if k=="primary": np.testing.assert_array_equal(c[:,1],prior[k][:8,1])
                    audit.append([seed,arm,k,d.tolist()])
    (OUT/"audit.json").write_text(json.dumps(dict(all_saved_counts_identical=True,independent_volumes=8,
        absolute_count_differences=audit),indent=2)+"\n")
    print("All saved-weight counts reproduced. Independent primary exact counts match.")


if __name__=="__main__":main()
