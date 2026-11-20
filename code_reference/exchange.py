"""Interchange the putative read-only lane across paired executable programs."""
import json
import string
import random
import numpy as np
import mlx.core as mx
from train_code import CodeLM, OUT, RUNS, write

def pairs(count):
    for i in range(count):
        rng=random.Random(f"cpython-ref-swap-20261119-{i}")
        stem="item_"+"".join(rng.choices(string.ascii_lowercase,k=5))
        a,b=rng.sample(string.ascii_lowercase,2)
        gap="".join(f"    tmp{j} = {rng.randrange(10,100)}\n"
                    for j in range(rng.randint(8,14)))
        # The generated constants have fixed width and no leading zeros.
        prefix="def fetch():\n    "+stem
        suffix=" = 7\n"+gap+"    return "+stem
        prompts=[prefix+c+suffix for c in (a,b)]
        for p,c in zip(prompts,(a,b)):
            scope={};exec(compile(p+c+"\n","<generated fetch>","exec"),scope)
            assert scope["fetch"]()==7
        yield prompts, (ord(a),ord(b))

def logits(model,h,idx):
    x=model.blocks[-1](h,False)
    y=model.read(model.norm(x[mx.arange(len(idx)),mx.array(idx)]).astype(x.dtype)).astype(mx.float32)
    return np.array(y)

def main():
    batch=list(pairs(1024));details={}
    for arm in ("fixed","writable"):
        model=CodeLM()
        data=mx.load(str(RUNS/"cpython-reference-main"/(arm+".safetensors")))
        model.load_weights([(k[8:],v) for k,v in data.items() if k.startswith("weights.")])
        assert int(data["step"].item())==16384
        del data
        scores={m:[] for m in ("clean","designated","neighbor","whole")}
        for offset in range(0,len(batch),8):
            cases=batch[offset:offset+8]
            prompts=[pair[member] for member in (0,1) for pair,_ in cases]
            gold=np.array([q[member] for member in (0,1) for _,q in cases])
            size=len(prompts)//2
            ix=np.r_[size:2*size,0:size]
            lengths=np.array([len(p) for p in prompts])
            ids=np.zeros((len(prompts),int(lengths.max())),np.int32)
            for n,p in enumerate(prompts):ids[n,:len(p)]=list(p.encode("ascii"))
            x=(model.tokens(mx.array(ids))+model.pos(mx.arange(ids.shape[1]))).astype(mx.bfloat16)
            for j,b in enumerate(model.blocks[:-1]):x=b(x,arm=="fixed" and 2<=j<9)
            other=x[mx.array(ix)]
            hidden={"clean":x,"designated":mx.concatenate([x[...,:448],other[...,448:]],axis=-1),
                "neighbor":mx.concatenate([x[...,:384],other[...,384:448],x[...,448:]],axis=-1),"whole":other}
            z={k:logits(model,h,lengths-1) for k,h in hidden.items()}
            np.testing.assert_array_equal(z["clean"][ix],z["whole"])
            for k,v in z.items():
                margin=v[np.arange(len(v)),gold]-v[np.arange(len(v)),gold[ix]]
                rows=np.stack([gold,v.argmax(1),margin],axis=1).reshape(2,size,3).transpose(1,0,2)
                scores[k].append(rows)
        arr={k:np.concatenate(v) for k,v in scores.items()}
        np.savez_compressed(OUT/(arm+"-exchange.npz"),**arr)
        both={k:(a[:,:,2]>0).all(1) for k,a in arr.items()}
        diff=(both["clean"]&~both["designated"]).astype(int)-(both["clean"]&~both["neighbor"]).astype(int)
        picks=np.random.default_rng(7851).integers(len(diff),size=(4000,len(diff)))
        ci=np.quantile(diff[picks].mean(axis=1),[.025,.975]).tolist()
        details[arm]=dict(modes={k:dict(pairs=int(both[k].sum()),n=len(v),top1=int((v[:,:,0]==v[:,:,1]).sum()),
            mean_log_odds=float(v[:,:,2].mean()),lost=int((both["clean"]&~both[k]).sum()),
            gained=int((~both["clean"]&both[k]).sum())) for k,v in arr.items()},
            extra_designated_losses=int(diff.sum()),interval=ci,qualified=bool(both["clean"].mean()>.50 and diff.sum()>=32 and ci[0]>0))
    write("result.json",dict(decision="complete",models=details,pairs=len(batch),seed_scheme="cpython-ref-swap-20261119"))
if __name__=="__main__":main()

