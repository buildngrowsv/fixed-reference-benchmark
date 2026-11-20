"""Read and interchange the fixed slice at the byte and block that produce it."""
import mlx.core as mx
import numpy as np

from exchange import pairs
from retrofit import states
from train_code import CodeLM, OUT, RUNS, write

WIDTH, LANE = 1536, 192
SLICES = {"reference": slice(WIDTH-LANE, WIDTH), "neighbor": slice(WIDTH-2*LANE, WIDTH-LANE)}
CHECKPOINT = "large-python-width-rate-main"


def batches(model, count, first):
    cases = list(pairs(count, start=first))
    for i in range(0, count, 8):
        rows = cases[i:i+8]
        prompts = [prompt for pair, _ in rows for prompt in pair]
        y = np.array([code for _, codes in rows for code in codes])
        lengths = np.array([len(p) for p in prompts])
        partner = np.arange(len(y)) ^ 1
        ids = np.zeros((len(y), lengths.max()), np.int32)
        sites = []
        for a,b in zip(prompts[::2],prompts[1::2],strict=True):
            diff = [j for j,(x,z) in enumerate(zip(a,b,strict=True)) if x!=z]
            assert len(diff)==1
            sites.append(diff[0])
        site = np.repeat(sites,2)
        for j,p in enumerate(prompts):
            ids[j,:len(p)] = np.frombuffer(p.encode("ascii"),np.uint8)
        assert np.array_equal(ids[np.arange(len(y)),site],y)
        hidden = states(model,mx.array(ids))
        mx.eval(hidden)
        yield hidden,site,lengths-1,y,partner


def fit_decoder(x, codes):
    labels = np.eye(26)[codes-ord("a")]
    center = x.mean(0)
    balance = labels.mean(0)
    c = x-center
    gram = c.T@c
    penalty = .02*np.trace(gram)/len(gram)
    gram += penalty*np.eye(len(gram))
    w = np.linalg.solve(gram,c.T@(labels-balance))
    return w,balance-center@w


def predict(model, hidden, ends):
    for j,block in enumerate(model.blocks[2:],start=2):
        hidden = block(hidden,j<9)
    x = hidden[mx.arange(len(ends)),mx.array(ends)]
    return np.array(model.read(model.norm(x).astype(x.dtype)).astype(mx.float32))


def margin(scores, y, partner):
    idx = np.arange(len(y))
    return (scores[idx,y]-scores[idx,y[partner]]).reshape(-1,2)


def confidence(v):
    rng = np.random.default_rng(59472)
    sampled = rng.integers(len(v),size=(4000,len(v)))
    return np.quantile(v[sampled].mean(1),[.025,.975]).tolist()


def main():
    mx.set_cache_limit(3*1024**3)
    model = CodeLM(WIDTH,LANE)
    data = mx.load(str(RUNS/CHECKPOINT/"fixed.safetensors"))
    assert data["step"].item()==16384
    model.load_weights([(k[8:],v) for k,v in data.items() if k.startswith("weights.")])
    del data
    design, labels = [], []
    for h,site,_,y,_ in batches(model,1024,16000):
        design.append(np.array(h[mx.arange(len(y)),mx.array(site)].astype(mx.float32)))
        labels.append(y)
    norms = np.concatenate(design).astype(np.float64)
    codes = np.concatenate(labels)
    readouts = {name:fit_decoder(norms[:,span],codes) for name,span in SLICES.items()}
    raw = {k:[] for k in ("native","reference","neighbor","whole_site","whole_state","readout_reference","readout_neighbor")}
    gold = []
    for h,site,end,y,partner in batches(model,512,18000):
        rows, locations = mx.arange(len(y)),mx.array(site)
        source = h[rows,locations]
        borrowed = source[mx.array(partner)]
        position = (mx.arange(h.shape[1])[None,:] == locations[:,None])[...,None]
        native = predict(model,h,end)
        whole = predict(model,h[mx.array(partner)],end)
        np.testing.assert_array_equal(whole,native[partner])
        raw["native"].extend(margin(native,y,partner))
        raw["whole_state"].extend(margin(whole,y,partner))
        states_at_site = {"whole_site":borrowed}
        for name,span in SLICES.items():
            columns = mx.array(np.isin(np.arange(WIDTH),np.arange(WIDTH)[span]))
            states_at_site[name] = mx.where(columns,borrowed,source)
            w,b = readouts[name]
            z = np.array(source[:,span].astype(mx.float32)) @ w+b
            raw["readout_"+name].extend(margin(z,y-ord("a"),partner))
        for name,contents in states_at_site.items():
            p = predict(model,mx.where(position,contents[:,None,:],h),end)
            raw[name].extend(margin(p,y,partner))
        gold.extend(y.reshape(-1,2))
    scores = {k:np.asarray(v) for k,v in raw.items()}
    clean = (scores["native"]>0).all(1)
    switches = {k:clean & (scores[k]<0).all(1) for k in ("reference","neighbor","whole_site","whole_state")}
    delta = switches["reference"].astype(int)-switches["neighbor"]
    ci = confidence(delta)
    recovered = {k:int((scores["readout_"+k]>0).all(1).sum()) for k in SLICES}
    qualifies = clean.mean()>=.5 and delta.mean()>=.05 and ci[0]>0
    readable_low_use = recovered["reference"]>=.8*len(clean) and switches["whole_site"].mean()>=.2 and switches["reference"].mean()<.05
    np.savez_compressed(OUT/"margins.npz",gold=np.asarray(gold),**scores)
    np.savez_compressed(OUT/"readouts.npz",**{name+"_"+part:z for name,values in readouts.items()
                        for part,z in zip(("weight","intercept"),values)})
    write("result.json",dict(decision="advance" if qualifies else "complete",test_pairs=len(clean),clean=int(clean.sum()),
           checkpoint=CHECKPOINT, readout_correct=recovered, switches={k:int(v.sum()) for k,v in switches.items()},
           fixed_minus_neighbor=float(delta.mean()),interval=ci,readable_low_use=bool(readable_low_use)))


if __name__ == "__main__":
    main()
