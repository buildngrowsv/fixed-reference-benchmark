"""A paired edit objective for the existing convolutional and recurrent hosts."""
from functools import partial
from pathlib import Path
import hashlib
import time

import chain_transfer as chain
from learned_address import OUT, atomic_json, ci, mx, nn, np, optim
from unfreeze_chain import fingerprint, route_arrays

ROOT=Path(__file__).resolve().parent/"chain_study"
ARMS=("donor","scratch","pretrained")
REGIMES=("clean","paired")
MARGIN=.01


def tune(model, seed, arm):
    opt=optim.Adam(.003,bias_correction=True)

    def objective(m, clean, current, stale, gold, new):
        encoded=m.encoder(m.embedding(clean[5]-m.offset))
        weights=mx.concatenate(tuple(m.weights(p) for p in (clean,current,stale)),axis=1)
        main,edit,control=mx.split(m.output(weights@encoded),[16,18],axis=1)
        old=gold[:,:2]
        def ce(logits, labels):
            return nn.losses.cross_entropy(logits,labels,reduction="mean")
        def choice(logits, labels):
            return mx.take_along_axis(logits,labels[...,None],axis=-1)[...,0]
        rank=(nn.softplus(1+choice(edit,old)-choice(edit,new)).mean()
             +nn.softplus(1+choice(control,new)-choice(control,old)).mean())*.5
        return .6*ce(main,gold)+.2*ce(edit,new)+.2*ce(control,old)+.1*rank

    gradient=nn.value_and_grad(model,objective)
    state=[model.state,opt.state]

    @partial(mx.compile,inputs=state,outputs=state)
    def step(clean,current,stale,gold,new):
        loss,grad=gradient(model,clean,current,stale,gold,new)
        opt.update(model,grad)
        return loss

    start=time.monotonic()
    for i in range(1,chain.TARGET_STEPS+1):
        clean=chain.program([seed,i],chain.BATCH,12,model.offset)
        current,stale=chain.counterfactuals(clean)
        gold=chain.oracle(clean)[2]
        new=chain.oracle(current)[2]
        loss=step(*(chain.to_mx(p) for p in (clean,current,stale)),
                  mx.array(gold-model.offset,dtype=mx.int32),mx.array(new-model.offset,dtype=mx.int32))
        mx.eval(loss,state)
        if i==1 or i%32==0:
            if not np.isfinite(loss.item()):
                raise ValueError(f"nonfinite loss: {arm}, update {i}")
            atomic_json("progress.json",dict(arm=arm,step=i,target=chain.TARGET_STEPS,loss=loss.item(),time=time.time()))
    return time.monotonic()-start


def panel(models, seed, offset, inputs=False):
    rows=[]
    for j in range(16):
        clean=chain.program([seed,j],chain.BATCH,20,offset)
        current,stale=chain.counterfactuals(clean)
        data={}
        for name,p in (("clean",clean),("current",current),("stale",stale)):
            _,loc,gold=chain.oracle(p)
            data.update({name+"_gold":gold,name+"_source":loc})
            for k,m in models.items():
                pos,pred=chain.predict_host(m,p)
                data.update({k+"_"+name:pred,k+"_"+name+"_source":pos})
        if inputs:
            data.update({"input_"+name:arr for name,arr in zip(chain.Records._fields,clean)})
            data.update(current_link=current.a_links,stale_link=stale.a_links)
        rows.append(data)
    return {k:np.concatenate([a[k] for a in rows]).astype(np.uint8) for k in rows[0]}


def ratio_interval(hit,eligible):
    rng=np.random.default_rng(771921)
    good=(hit&eligible).sum(1)
    count=eligible.sum(1)
    draws=np.concatenate([rng.integers(len(hit),size=(200,len(hit))) for _ in range(20)])
    matches=good[draws].sum(1)
    n=count[draws].sum(1)
    return np.quantile(np.divide(matches,n,out=np.zeros(len(n)),where=n>0),[.025,.975]).tolist()


def metric(predictions):
    scores={}
    acc={}
    for a in ARMS:
        for regime in REGIMES:
            k=a+"_"+regime
            clean=predictions[k+"_clean"]==predictions["clean_gold"]
            acc[k]=clean.mean(1)
            s=dict(clean=float(clean.mean()),edits={})
            good=clean[:,:2]
            for e in ("current","stale"):
                ok=predictions[k+"_"+e]==predictions[e+"_gold"]
                valid=int(good.sum())
                both=int((good&ok).sum())
                s["edits"][e]=dict(conditional=both/valid if valid else None, clean_correct=valid,
                    both_correct=both,unconditional=float(ok.mean()),edit_correct=int(ok.sum()),
                    interval=ratio_interval(ok,good))
            scores[k]=s
    changes={k:dict(delta=float(d.mean()),interval=ci(d)) for k,d in
        ((a,acc[a+"_paired"]-acc[a+"_clean"]) for a in ARMS)}
    relative={k:dict(delta=float(d.mean()),interval=ci(d)) for k,d in
        ((a,acc["donor_paired"]-acc[a+"_paired"]) for a in ("scratch","pretrained"))}
    donor=scores["donor_paired"]
    return dict(scores=scores,changes=changes,donor_vs_paired_native=relative,
        qualifies=bool(changes["donor"]["interval"][0]>-MARGIN and
            all((m["conditional"] or 0)>=.9 for m in donor["edits"].values())))


def verify_replay(p,label):
    base=ROOT/"results"/(label+".npz")
    prior=ROOT/"adaptation/results"/(label+".npz")
    with np.load(base) as b,np.load(prior) as a:
        for arm,expected in (("donor",a["graft_answer"]),("scratch",b["scratch_answer"]),
                             ("pretrained",b["pretrained_answer"])):
            np.testing.assert_array_equal(p[arm+"_clean_clean"],expected)
        for edit in ("current","stale"):
            np.testing.assert_array_equal(p[edit+"_gold"],b[edit+"_gold"])
            np.testing.assert_array_equal(p["donor_clean_"+edit],a[edit+"_answer"])
        np.testing.assert_array_equal(p["clean_gold"],b["gold"])
    return {path.parent.parent.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in (base,prior)}


def main():
    mx.set_default_device(mx.cpu)
    mx.set_cache_limit(64*1024*1024)
    original=chain.source(101,"reconstruct seed-101 router")
    witness=chain.verify_donor(original)
    fits={}
    for seed,pretrain_seed in ((503,341),(607,861)):
        native=chain.source(pretrain_seed,f"{seed}: native one-hop pretraining")
        for host,encoder,offset in (("convolution",chain.Convolution,64),("recurrent",chain.Recurrent,128)):
            label=f"{host}-{seed}"
            models={}
            init={}
            for kind,source in (("donor",original),("scratch",None),("pretrained",native)):
                for regime in REGIMES:
                    mx.random.seed(seed)
                    m=chain.Host(encoder,offset)
                    if source is not None:
                        m.locator.update(source.locator.parameters())
                    k=kind+"_"+regime
                    init[k]=fingerprint(route_arrays(m))
                    if regime=="clean":
                        chain.train(m,42918+seed,f"{label}/{k}")
                    else:
                        tune(m,42918+seed,f"{label}/{k}")
                    models[k]=m
            assert all(init[a+"_clean"]==init[a+"_paired"] for a in ARMS)
            replay=panel(models,83639+seed,offset)
            hashes=verify_replay(replay,label)
            held=panel(models,719991+seed,offset,inputs=True)
            np.savez_compressed(OUT/(label+".npz"),**held)
            fits[label]=dict(held_out=metric(held),original_panel=metric(replay),archived_hashes=hashes,
                matched_initial_router_sha256={a:init[a+"_clean"] for a in ARMS})
    atomic_json("result.json",dict(decision="complete",qualifies=all(f["held_out"]["qualifies"] for f in fits.values()),
        fits=fits,clean_margin=MARGIN,steps=chain.TARGET_STEPS,device=str(mx.default_device()),
        hardware=mx.device_info()["device_name"],donor_witness_sha256=witness))


if __name__=="__main__":
    main()
