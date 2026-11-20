"""Supervise the final write, with a matched host and matched direct-read baselines."""
from functools import partial
import hashlib
from pathlib import Path
import time

import chain_transfer as chain
import paired_edits as paired
from learned_address import OUT, atomic_json, ci, mx, nn, np, optim
from unfreeze_chain import fingerprint, route_arrays

ARMS=("donor","scratch","pretrained")
REGIMES=("clean","answer_edits","write")
EDITS=("current","stale")
PHASES=("clean",*EDITS)
BASE=Path(__file__).resolve().parent/"chain_study/paired_edits/results"
CLEAN_MARGIN=.01


def fit_write(model, seed, arm):
    opt=optim.Adam(.003,bias_correction=True)

    def objective(m, programs, positions, answers):
        encoded=m.encoder(m.embedding(programs[0][5]-m.offset))
        route=tuple(m.weights(p) for p in programs)
        def location_loss(prob,target):
            chosen=mx.take_along_axis(prob,target[...,None],axis=-1)[...,0]
            return -mx.log(mx.maximum(chosen,1.e-9)).mean()
        pointer=sum(w*location_loss(prob,target) for w,prob,target in zip((.6,.2,.2),route,positions))
        logits=mx.split(m.output(mx.concatenate(route,axis=1)@encoded),[16,18],axis=1)
        payload=sum(w*nn.losses.cross_entropy(z,y,reduction="mean")
                    for w,z,y in zip((.6,.2,.2),logits,answers))
        return pointer+.25*payload

    grad=nn.value_and_grad(model,objective)
    state=[model.state,opt.state]

    @partial(mx.compile,inputs=state,outputs=state)
    def step(programs,positions,answers):
        loss,g=grad(model,programs,positions,answers)
        opt.update(model,g)
        return loss

    for i in range(1,chain.TARGET_STEPS+1):
        base=chain.program([seed,i],chain.BATCH,12,model.offset)
        programs=(base,*chain.counterfactuals(base))
        gold=[chain.oracle(p)[1:] for p in programs]
        pos=tuple(mx.array(v,dtype=mx.int32) for v,_ in gold)
        values=tuple(mx.array(v-model.offset,dtype=mx.int32) for _,v in gold)
        loss=step(tuple(chain.to_mx(p) for p in programs),pos,values)
        mx.eval(loss,state)
        if i==1 or i%32==0:
            assert np.isfinite(loss.item())
            atomic_json("progress.json",dict(arm=arm,step=i,target=chain.TARGET_STEPS,loss=loss.item(),time=time.time()))


def panel(models, seed, offset, count, record_inputs=False):
    rows=[]
    for j in range(count//chain.BATCH):
        clean=chain.program([seed,j],chain.BATCH,20,offset)
        current,stale=chain.counterfactuals(clean)
        data={}
        for name,p in zip(PHASES,(clean,current,stale)):
            _,position,gold=chain.oracle(p)
            data[name+"_gold"]=gold
            data[name+"_source"]=position
            for k,m in models.items():
                pos,soft=chain.predict_host(m,p)
                data[k+"_"+name+"_source"]=pos
                if name=="clean" or not record_inputs:
                    data[k+"_"+name+"_soft"]=soft
        if record_inputs:
            data.update({"input_"+key:v for key,v in zip(chain.Records._fields,clean)})
            data.update(current_link=current.a_links,stale_link=stale.a_links)
        rows.append(data)
    return {key:np.concatenate([r[key] for r in rows]).astype(np.uint8) for key in rows[0]}


def old_panel(models, seed, offset, label):
    names=[a+"_"+r for a in ARMS for r in REGIMES[:2]]
    replay=panel({k:models[k] for k in names},719991+seed,offset,1024)
    archive=BASE/(label+".npz")
    with np.load(archive) as reference:
        for phase in PHASES:
            for arm in ARMS:
                for regime,old in (("clean","clean"),("answer_edits","paired")):
                    new=arm+"_"+regime+"_"+phase
                    past=arm+"_"+old+"_"+phase
                    np.testing.assert_array_equal(replay[new+"_soft"],reference[past])
                    np.testing.assert_array_equal(replay[new+"_source"],reference[past+"_source"])
            np.testing.assert_array_equal(replay[phase+"_gold"],reference[phase+"_gold"])
    return hashlib.sha256(archive.read_bytes()).hexdigest()


def score(data):
    payloads=data["input_payloads"]
    terms={}
    values={}
    for a in ARMS:
        for r in REGIMES:
            name=a+"_"+r
            hits={}
            for p in PHASES:
                value=np.take_along_axis(payloads,data[name+"_"+p+"_source"],axis=1)
                hits[p]=value==data[p+"_gold"]
            values[name]=hits
            terms[name]={p:dict(accuracy=float(v.mean()),correct=int(v.sum()),n=v.size,
                                interval=ci(v.mean(1))) for p,v in hits.items()}
            terms[name]["both_edits"]=float((hits["current"]&hits["stale"]).mean())
            terms[name]["complete_edited_program"]=float((hits["current"]&hits["stale"]).all(1).mean())
            terms[name]["clean_neural_readout"]=float((data[name+"_clean_soft"]==data["clean_gold"]).mean())
    changes={}
    for a in ARMS:
        for r in REGIMES[:2]:
            changes[a+"_vs_"+r]={}
            for p in PHASES:
                d=values[a+"_write"][p].mean(1)-values[a+"_"+r][p].mean(1)
                changes[a+"_vs_"+r][p]=dict(delta=float(d.mean()),interval=ci(d))
    def threshold(a):
        d=changes[a+"_vs_clean"]["clean"]
        edits=[terms[a+"_write"][p] for p in EDITS]
        return dict(clears_both_edits=all(x["accuracy"]>=.9 for x in edits),
                    both_lower_bounds_above_90=all(x["interval"][0]>.90 for x in edits),
                    clean_held=d["delta"]>=0 and d["interval"][0]>-CLEAN_MARGIN)
    return dict(scores=terms,contrasts=changes,decision={a:threshold(a) for a in ARMS})


def main():
    mx.set_default_device(mx.cpu)
    mx.set_cache_limit(64*1024*1024)
    donor=chain.source(101,"reconstruct original donor")
    witness=chain.verify_donor(donor)
    results={}
    for seed,pretrain_seed in ((503,341),(607,861)):
        native=chain.source(pretrain_seed,f"{seed}: matched native pretraining")
        for host,arch,offset in (("convolution",chain.Convolution,64),("recurrent",chain.Recurrent,128)):
            label=f"{host}-{seed}"
            models={}
            initial={}
            for arm,initializer in (("donor",donor),("scratch",None),("pretrained",native)):
                for regime in REGIMES:
                    mx.random.seed(seed)
                    m=chain.Host(arch,offset)
                    if initializer is not None:
                        m.locator.update(initializer.locator.parameters())
                    key=arm+"_"+regime
                    initial[key]=fingerprint(route_arrays(m))
                    run={"clean":chain.train,"answer_edits":paired.tune,"write":fit_write}[regime]
                    run(m,42918+seed,f"{label}/{key}")
                    models[key]=m
            assert all(len({initial[a+"_"+r] for r in REGIMES})==1 for a in ARMS)
            checksum=old_panel(models,seed,offset,label)
            fresh=panel(models,968307+seed,offset,2048,record_inputs=True)
            np.savez_compressed(OUT/(label+".npz"),**fresh)
            results[label]=dict(primary=score(fresh),previous_panel_sha256=checksum,
                matched_initial_sha256={a:initial[a+"_clean"] for a in ARMS})
    flags=[r["primary"]["decision"]["donor"] for r in results.values()]
    atomic_json("result.json",dict(decision="complete",passes=all(f["clears_both_edits"] and f["clean_held"] for f in flags),
        strict_edit_intervals=all(f["both_lower_bounds_above_90"] for f in flags),
        fits=results,target_steps=chain.TARGET_STEPS,programs_per_fit=2048,clean_loss_margin=CLEAN_MARGIN,
        device=str(mx.default_device()),hardware=mx.device_info()["device_name"],donor_witness_sha256=witness))


if __name__=="__main__":
    main()
