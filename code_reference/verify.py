"""Reaggregate code-model scores and paired interventions without loading MLX."""
import json
from pathlib import Path
import sys
import numpy as np

def check(a,b):np.testing.assert_allclose(a,b,rtol=1.e-8,atol=1.e-10)

def grouped_difference(a,b,groups):
    keys,idx=np.unique(groups,return_inverse=True)
    sums=np.bincount(idx,weights=a-b)
    totals=np.bincount(idx)*512
    rng=np.random.default_rng(9651)
    dist=[]
    for _ in range(50):
        sampled=rng.choice(len(keys),(100,len(keys)),replace=True)
        dist.extend(sums[sampled].sum(1)/totals[sampled].sum(1))
    return dict(delta=float((a-b).mean()/512),interval=np.quantile(dist,[.025,.975]).tolist(),packages=len(keys))

def main(root):
    main=root/"main"
    summary=json.loads((main/"result.json").read_text())
    corpus=json.loads((main/"corpus.json").read_text())["files"]
    splits={k:{r["group"] for r in v} for k,v in corpus.items()}
    assert not (splits["train"]&splits["test"] or splits["train"]&splits["val"] or splits["test"]&splits["val"])
    report=dict(corpus={k:dict(files=len(v),bytes=sum(r["bytes"] for r in v),packages=len(splits[k])) for k,v in corpus.items()})
    for split,label,n in (("validation","val",1024),("test","test",2048)):
        arrays={}
        for arm in ("fixed","writable"):
            f=np.load(main/(arm+"-"+split+".npz"))
            v=f["scores"];rows=f["windows"];groups=f["groups"]
            assert v.shape==(n,5)
            for (i,off),g in zip(rows,groups):
                file=corpus[label][i]
                assert g==file["group"] and 0<=off<=file["bytes"]-513
            check(v[:,0].sum()/(n*512),summary["models"][arm][split+"_bpb"])
            if split=="test":
                check(v[:,1].sum()/(n*512),summary["models"][arm]["byte_accuracy"])
                check(v[:,3].sum()/v[:,2].sum(),summary["models"][arm]["repeated_identifier_bpb"])
            arrays[arm]=(v,rows,groups)
        for col in (1,2):np.testing.assert_array_equal(arrays["fixed"][col],arrays["writable"][col])
        report[split]=grouped_difference(arrays["fixed"][0][:,0],arrays["writable"][0][:,0],groups)
        if split=="validation":check(report[split]["interval"],summary["validation_interval"])
    exchange=json.loads((root/"exchange/result.json").read_text())
    for arm,h in exchange["models"].items():
        f=np.load(root/"exchange"/(arm+"-exchange.npz"));clean=(f["clean"][:,:,2]>0).all(1)
        for mode in h["modes"]:
            a=f[mode];both=(a[:,:,2]>0).all(1);m=h["modes"][mode]
            assert m["pairs"]==both.sum() and m["lost"]==(clean&~both).sum() and m["gained"]==(both&~clean).sum()
            assert m["top1"]==(a[:,:,0]==a[:,:,1]).sum()
        np.testing.assert_array_equal(f["whole"][:,:,1],f["clean"][:,::-1,1])
        np.testing.assert_array_equal(f["whole"][:,:,2],-f["clean"][:,::-1,2])
    shared=json.loads((root/"shared/result.json").read_text())
    a=np.load(root/"shared/shared-validation.npz");b=np.load(root/"shared/shallow-validation.npz")
    np.testing.assert_array_equal(a["windows"],b["windows"])
    report["shared_minus_shallow"]=grouped_difference(a["scores"][:,0],b["scores"][:,0],a["groups"])
    check(report["shared_minus_shallow"]["interval"],shared["contrasts"]["val"]["shallow"]["interval"])
    report["exchange_pairs_per_model"]=len(clean)
    print(json.dumps(report,indent=2))
if __name__=="__main__":main(Path(sys.argv[1]))
