"""Exchange only the variable-defining byte's carried state at the sealed exit."""
import sys
import numpy as np
from bypass import Bypass, mx, OUT, RUNS, write
from exchange import pairs, logits


def run(file, cases):
    model = Bypass()
    weights = mx.load(str(file))
    assert int(weights["step"].item()) == 16384
    model.load_weights([(k[8:], v) for k,v in weights.items() if k.startswith("weights.")])
    del weights
    records = {k:[] for k in ("clean","reference","working","full_site","whole")}
    for offset in range(0,len(cases),8):
        group = cases[offset:offset+8]
        prompts = [pair[j] for j in (0,1) for pair,_ in group]
        gold = np.array([labels[j] for j in (0,1) for _,labels in group])
        n = len(group)
        partner = np.r_[n:2*n, 0:n]
        lengths = np.array([len(s) for s in prompts])
        ids = np.zeros((2*n,max(lengths)),np.int32)
        for i,s in enumerate(prompts):
            ids[i,:len(s)] = list(s.encode("ascii"))
        changed = ids != ids[partner]
        assert np.all(changed.sum(1)==1)
        source = changed.argmax(1)
        np.testing.assert_array_equal(ids[np.arange(2*n),source], gold)
        tokens = mx.array(ids)
        x = (model.tokens(tokens)+model.pos(mx.arange(ids.shape[1]))).astype(mx.bfloat16)
        x = model.blocks[1](model.blocks[0](x,False),False)
        x, reference = mx.split(x,[448],axis=-1)
        for block in model.blocks[2:9]:
            x = block(x,False)
        h = mx.concatenate([x,reference],axis=-1)
        other = h[mx.array(partner)]
        site = mx.array(changed)[...,None]
        changed_site = mx.where(site,other,h)
        states = dict(clean=h, reference=mx.concatenate([x,changed_site[...,448:]],axis=-1),
                      working=mx.concatenate([x[...,:384],changed_site[...,384:448],reference],axis=-1),
                      full_site=changed_site,whole=other)
        outcomes = {k:logits(model,v,lengths-1) for k,v in states.items()}
        np.testing.assert_array_equal(outcomes["whole"], outcomes["clean"][partner])
        if offset==0:
            clean_all = model(tokens)
            np.testing.assert_array_equal(outcomes["clean"], np.array(clean_all[mx.arange(2*n),mx.array(lengths-1)]))
        for k,z in outcomes.items():
            margin = z[np.arange(2*n),gold] - z[np.arange(2*n),gold[partner]]
            rows = np.stack([gold,z.argmax(1),margin],axis=1)
            records[k].append(rows.reshape(2,n,3).transpose(1,0,2))
    return {k:np.concatenate(v) for k,v in records.items()}


def summarize(arr):
    baseline = arr["clean"]
    good = (baseline[:,:,2]>0).all(1)
    both_top = (baseline[:,:,1]==baseline[:,:,0]).all(1)
    def reversed_preference(k):
        return good & (arr[k][:,:,2]<0).all(1)
    d = reversed_preference("reference").astype(float)-reversed_preference("working")
    rng = np.random.default_rng(393217)
    draws = np.concatenate([d[rng.integers(len(d),size=(100,len(d)))].mean(1) for _ in range(40)])
    ci = np.quantile(draws,[.025,.975]).tolist()
    return dict(clean_pairs=int(good.sum()),clean_top_one_pairs=int(both_top.sum()),
                double_reversals={k:int(reversed_preference(k).sum()) for k in arr},
                exact_partner_answers={k:int((both_top & (a[:,:,1]==baseline[:,::-1,0]).all(1)).sum()) for k,a in arr.items()},
                difference=float(d.mean()),interval=ci,meets_threshold=bool(good.mean()>=.5 and d.mean()>=.03 and ci[0]>0))


def main():
    smoke = sys.argv[1:]==["smoke"]
    if smoke:
        mx.set_default_device(mx.cpu)
    cases = list(pairs(1 if smoke else 1024,start=4096))
    models = [(101, RUNS/"sealed-bypass-main/bypass.safetensors"),(211,RUNS/"sealed-bypass-repeat/sealed.safetensors")]
    report = {}
    for seed,file in models:
        arr = run(file,cases)
        if smoke:
            print("Clean versus executable and swapped partner identically verified",seed)
            continue
        np.savez_compressed(OUT/("seed-"+str(seed)+".npz"),**arr)
        report[str(seed)] = summarize(arr)
    if not smoke:
        write("result.json",dict(decision="complete",seeds=report,pairs=len(cases),
              replicates=all(v["meets_threshold"] for v in report.values())))


if __name__=="__main__":
    main()
