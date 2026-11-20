"""Independent NumPy forward pass over all published receiver checkpoints."""
from itertools import product
import json
from pathlib import Path
import sys
import numpy as np
from safetensors.numpy import load_file

def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-80,80)))
def neural(x,w,prefix):
    h=np.maximum(0,x@w[prefix+"hidden.weight"].T+w[prefix+"hidden.bias"])
    return sigmoid(h@w[prefix+"output.weight"].T+w[prefix+"output.bias"])
def rule(x):
    return (x[...,0]==x[...,3]) | ((x[...,1]==1)&(x[...,4]==1))
def main():
    folder=Path(sys.argv[1])
    donor=load_file(str(folder/"donor-101.safetensors"))
    raw=load_file(sys.argv[2])
    for k,v in raw.items():np.testing.assert_array_equal(v,donor["donor."+k])
    table=np.array(list(product(range(2),repeat=6)),np.float32)
    labels=rule(table).astype(np.float32)
    corrupted=np.random.default_rng(7612).permutation(labels)
    assert np.array_equal(neural(table,raw,"")[:,0]>.5,labels)
    saved=np.load(folder/"predictions.npz")
    report={}
    for split,n,length,seed in (("val",512,8,3243),("test",2048,16,15274)):
        rng=np.random.default_rng(seed)
        bits=rng.integers(0,2,(n,length,6)).astype(np.float32)
        words=rng.choice(np.array([-1,1],np.float32),(n,length+1,32))
        bits[:,-1,[0,3,1,4]]=np.array([0,1,1,1])
        comparison={}
        for arm in ("donor","fresh","corrupted","analytic","donor_disabled","donor_corrupted","correct_wiring"):
            w=load_file(str(folder/((arm.split("_")[0] if arm!="correct_wiring" else "donor")+"-101.safetensors")))
            ok=np.ones(n,bool)
            for value in (1,0):
                x=bits.copy();x[:,-1,4]=value
                src=x[:,:,[3,0,5,2,1,4]]
                if arm=="fresh":g=neural(src,w,"native.")
                elif arm=="correct_wiring":g=neural(x,raw,"")
                elif arm=="donor_disabled":g=np.zeros_like(x[:,:,:1])
                else:
                    y=sigmoid(6*(src@w["adapter.weight"].T+w["adapter.bias"]-.5))
                    if arm=="donor":g=neural(y,w,"donor.")
                    else:
                        prob=np.prod(np.where(table==1,y[:,:,None,:],1-y[:,:,None,:]),axis=-1)
                        g=(prob@(corrupted if "corrupted" in arm else labels))[:,:,None]
                mem=words[:,0].copy();expected=mem.copy()
                for t in range(length):
                    sel=rule(x[:,t]);expected[sel]=words[sel,t+1]
                    mem+=g[:,t]*(words[:,t+1]-mem)
                ok&=((mem>0)==(expected>0)).all(1)
            if arm!="correct_wiring":np.testing.assert_array_equal(ok,saved[f"101_{split}_{arm}"])
            comparison[arm]=int(ok.sum())
        report[split]=comparison
    print(json.dumps(dict(source_table_correct=64,unchanged_donor_weights=True,individual_predictions_reproduced=True,panels=report),indent=2))
if __name__=="__main__":main()

