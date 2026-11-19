# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx==0.32.4", "numpy==2.5.5"]
# ///
"""Independent follow-up. Run prepare, pilot, run, then summarize on the Studio."""
import argparse
import hashlib
import json
import time
from functools import partial
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import benchmark as b


ROOT = b.ROOT / "balanced"
DATA = ROOT / "data"


def load(split):
    with np.load(DATA / (split + ".npz")) as f:
        return dict(f)


def prepare():
    DATA.mkdir(parents=True,exist_ok=True)
    seen = set()
    for name in ("train","val","test","sensitive"):
        seen.update(b.keys(b.load(name)).tolist())
    starting = len(seen)
    for split,n,seed in [("train",50000,30011),("val",5000,30013),("val_bal",2400,30023),
                         ("test",10000,30029),("sensitive",10000,30047)]:
        rng = np.random.default_rng(seed)
        balanced = split in ("val_bal","sensitive")
        counts = np.zeros((2,b.COLORS),dtype=int)
        chunks,used = [],0
        while used < n:
            data=b.examples(rng,12000)
            pick=[]
            for i,key in enumerate(b.keys(data).tolist()):
                st=int(data["start"][i]!=data["final"][i]), int(data["final"][i])
                if key in seen or (balanced and (not data["sensitive"][i] or counts[st]>=n//8)):
                    continue
                pick.append(i)
                seen.add(key)
                counts[st]+=1
                if used+len(pick)==n:
                    break
            if pick:
                chunks.append({k:v[pick] for k,v in data.items()})
                used+=len(pick)
        joined = {k:np.concatenate([p[k] for p in chunks]) for k in chunks[0]}
        if balanced:
            assert np.all(counts==n//8)
        np.savez(DATA/(split+".npz"), **joined)
        print(split, n, counts.tolist(),flush=True)
    assert len(seen)-starting==77400
    b.check_networks(64)


def train(variant,seed,width,epochs,out,pilot=False):
    out.mkdir(parents=True,exist_ok=True)
    net=b.BoardNet(width,variant,seed)
    opt=optim.Adam(learning_rate=.001)
    def obj(model,x,c,original,current):
        # Model receives no query before the readout. Reuse its same forward
        # calculation for the eight possible square queries on the training board.
        logits=mx.stack([model(x,c,mx.full((x.shape[0],),j,mx.int32)) for j in range(b.SLOTS)],axis=2)
        old=nn.losses.cross_entropy(logits[:,0],original,reduction="mean")
        new=nn.losses.cross_entropy(logits[:,1],current,reduction="none")
        w=mx.where(current!=original,1.5,.75)
        return .25*old + .75*mx.mean(w*new)
    loss_grad=nn.value_and_grad(net,obj)
    state=[net.state,opt.state]
    @partial(mx.compile,inputs=state,outputs=state)
    def update(x,c,old,new):
        loss,grad=loss_grad(net,x,c,old,new)
        opt.update(net,grad)
        return loss
    tr=load("train")
    val, hard=load("val"),load("val_bal")
    x,c=b.boards(tr["board"]),mx.array(tr["cmd"])
    old=mx.array(tr["board"])
    final=np.take_along_axis(tr["board"],b.trace_sources(tr["cmd"]),axis=1)
    assert np.array_equal(final[np.arange(len(final)), tr["q"]],tr["final"])
    new=mx.array(final)
    mx.eval(x,c,old,new,net.parameters(),opt.state)
    rng=np.random.default_rng(50000+seed)
    curve=[]
    train_seconds=0
    mx.reset_peak_memory()
    for ep in range(1,epochs+1):
        t=time.perf_counter()
        perm=mx.array(rng.permutation(len(final)).astype(np.int32))
        for off in range(0,len(final),b.BATCH):
            idx=perm[off:off+b.BATCH]
            loss=update(x[idx],c[idx],old[idx],new[idx])
            mx.eval(loss,state)
        train_seconds+=time.perf_counter()-t
        vm=b.evaluate(net,val)[0]
        vm["balanced_restore"]=b.evaluate(net,hard)[0]["current"]
        curve.append(vm)
        if pilot or ep in (1,epochs):
            print(f"{variant} {seed} epoch {ep} val {vm['current']:.3f} balanced restore {vm['balanced_restore']:.3f}",flush=True)
    info=dict(variant=variant,seed=seed,width=width,epochs=epochs,batch=b.BATCH,
              examples_presented=len(final)*epochs,validation=curve,training_seconds=train_seconds,
              parameters=sum(v.size for _,v in tree_flatten(net.trainable_parameters())),
              train_peak_mib=mx.get_peak_memory()/2**20)
    if not pilot:
        preds={}
        for split in ("test","sensitive"):
            metrics,pred,extra=b.evaluate(net,load(split),patch=variant=="fixed" and split=="sensitive")
            info[split]=metrics
            preds[split]=pred
            preds.update(extra)
        np.savez_compressed(out/"predictions.npz",**preds)
    else:
        eligible=[2,4,8,12,16,24,32]
        # Validation only. Choose the earliest milestone within half a point
        # of the best equally weighted natural and balanced current accuracy.
        scores={i:.5*(curve[i-1]["current"]+curve[i-1]["balanced_restore"]) for i in eligible}
        info["chosen_epochs"]=next(i for i in eligible if scores[i]>=max(scores.values())-.005)
    b.write_json(out/"result.json",info)
    return info


def main(args):
    mx.set_default_device(mx.gpu)
    area=ROOT/f"width-{args.width}"
    if args.mode=="prepare":
        prepare()
        b.write_json(ROOT/"environment.json",dict(device=mx.device_info()["device_name"],
            base_sha256=hashlib.sha256(Path(b.__file__).read_bytes()).hexdigest(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    elif args.mode=="pilot":
        res=train("ordinary",37,args.width,32,area/"pilot",True)
        print("Locked common epoch budget",res["chosen_epochs"],flush=True)
    elif args.mode=="run":
        budget=json.loads((ROOT/"width-64"/"pilot"/"result.json").read_text())["chosen_epochs"]
        for i,seed in enumerate(b.SEEDS):
            for v in b.VARIANTS[i%4:]+b.VARIANTS[:i%4]:
                out=area/f"{seed}-{v}"
                if (out/"result.json").exists() and (out/"predictions.npz").exists():
                    continue
                result=train(v,seed,args.width,budget,out)
                print(f"DONE {v} {seed}: {result['sensitive']['current']:.4f}",flush=True)
                mx.clear_cache()
    else:
        from analyze import summarize
        summarize(ROOT,area,b.SEEDS,b.VARIANTS)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["prepare","pilot","run","summarize"])
    p.add_argument("--width",type=int,default=64)
    main(p.parse_args())
