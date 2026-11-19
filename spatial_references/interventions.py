"""Fresh-panel checks of write stability and crossing a single architectural doorway."""
import json
import mlx.core as mx
import numpy as np
from run import Grid, R, SEEDS, count, evaluate, examples, fit


def restore(arm,seed):
    record=json.loads((R/f"{arm}-{seed}.json").read_text())
    net=Grid(arm)
    net.read=mx.array(record["read"],dtype=mx.float32)
    if arm=="writable": net.write=mx.array(record["write"],dtype=mx.float32)
    return net


def door_test(net):
    counts={k:[] for k in ("open","closed","unrestricted_on_open","changed","different_predictions")}
    rng=np.random.default_rng(99171)
    for _ in range(64):
        s=int(rng.integers(2**31))
        x,a,opened,_=examples(np.random.default_rng(s),16,6,24,True)
        _,_,closed,_=examples(np.random.default_rng(s),16,6,24,False)
        mask=(opened!=closed).any(-1)
        xx,aa=mx.array(x),mx.array(a)
        o=np.array(net(xx,aa,door=True));c=np.array(net(xx,aa,door=False))
        p=np.array(net(xx,aa))
        for key,z,y in (("open",o,opened),("closed",c,closed),("unrestricted_on_open",p,opened)):
            counts[key].append(count(z,y,mask))
        counts["changed"].append(mask.sum((1,2)))
        counts["different_predictions"].append((((o>0)!=(c>0)).any(-1)&mask).sum((1,2)))
    return {k:np.concatenate(v) for k,v in counts.items()}


def main():
    for seed in SEEDS:
        fixed=restore("fixed",seed)
        writable=restore("writable",seed)
        convex,history=fit("convex",seed,400)
        out={}
        for k,net in (("fixed",fixed),("writable",writable),("convex",convex)):
            out[k]=evaluate(net,6,24,94531,batches=64)["normal"]
        writable.write=mx.zeros_like(writable.write)
        out["writes_off"]=evaluate(writable,6,24,94531,batches=64)["normal"]
        for k,v in door_test(fixed).items(): out["door_"+k]=v
        np.savez_compressed(R/f"intervention-{seed}.npz",**out)
        (R/f"convex-{seed}.json").write_text(json.dumps(dict(write=np.array(convex.write).tolist(),
                    read=np.array(convex.read).tolist(),loss=history),indent=2)+"\n")
        print(seed,{k:v.sum(0).tolist() for k,v in out.items()},flush=True)


if __name__=="__main__":
    main()
