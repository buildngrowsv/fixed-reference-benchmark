"""Matched validation gate and fresh-panel comparison of spatial bank initializations."""
import json
from pathlib import Path
import sys

import mlx.core as mx
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"learned_banks"))
from experiment import Assigned, Banks, R as PRIOR, accuracy, evaluate, panel, save_json, train

OUT=HERE/"results"
ARMS=("assigned","neutral","persistent","biased")


def initialize(m):
    bias=np.zeros((8,2,2,14),np.float32)
    for dest in range(2): bias[:,dest,:,7*dest]=np.log(117.)
    m.table=m.table+mx.array(bias)


def load_or_fit(arm,seed):
    own=OUT/f"{arm}-{seed}.json"
    old=PRIOR/("pilot.json" if arm=="biased" and seed==101 else f"{arm}-{seed}.json")
    if own.exists() or old.exists():
        record=json.loads((own if own.exists() else old).read_text())
        m=Assigned() if arm=="assigned" else Banks()
        m.load_weights([(k,mx.array(v,dtype=mx.float32)) for k,v in record["weights"].items()])
    else:
        m,record=train(arm,seed,initialize if arm=="persistent" else None)
    return m,record


def validation(seed):
    val=list(panel(371239,4,8,8))
    records={}
    for arm in ARMS:
        m,r=load_or_fit(arm,seed)
        ev=evaluate(m,arm,val)
        r["fresh_validation"]={k:accuracy(v) for k,v in ev.items()}
        r["fresh_address"]=int(np.argmax([accuracy(ev["probe0"]),accuracy(ev["probe1"])]))
        save_json(OUT/f"{arm}-{seed}.json",r)
        records[arm]=accuracy(ev["primary"])
    print("validation",seed,records,flush=True)
    return records


def main():
    OUT.mkdir(exist_ok=True)
    values=validation(101)
    qualified=values["persistent"]-values["neutral"]>=.05
    seeds=(101,211,307) if qualified else (101,)
    save_json(OUT/"decision.json",dict(validation=values,qualified=qualified,seeds=seeds))
    for seed in seeds[1:]: validation(seed)
    for seed in seeds:
        for arm in ARMS:
            m,r=load_or_fit(arm,seed)
            ev=evaluate(m,arm,panel(752916,6,16,16),True)
            np.savez_compressed(OUT/f"{arm}-{seed}.npz",**ev)
            print("test",seed,arm,accuracy(ev["primary"]),flush=True)


if __name__=="__main__": main()
