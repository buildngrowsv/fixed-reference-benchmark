"""Test a discrete inference path with the same scored locator at all three reads."""
import hashlib
from pathlib import Path

import chain_transfer as chain
import write_supervision as write
from learned_address import OUT,atomic_json,mx,np

BASE=Path(__file__).resolve().parent/"chain_study/write_supervision/results"
ARMS=write.ARMS


def discrete_prediction(model, program):
    a,al,b,bl,c,_,q,kind=chain.to_mx(program)
    router=model.locator
    row=mx.arange(len(q))[:,None]
    latest=mx.ones_like(kind)
    first=router(a,q,latest).argmax(-1)
    middle=router(b,al[row,first],latest).argmax(-1)
    finish=router(c,bl[row,middle],kind).argmax(-1)
    return np.array(finish)


def score_panel(models,offset,seed,collect_greedy):
    rows=[]
    for j in range(32):
        base=chain.program([seed,j],chain.BATCH,20,offset)
        programs=(base,*chain.counterfactuals(base))
        out={}
        for phase,p in zip(write.PHASES,programs):
            _,source,value=chain.oracle(p)
            out[phase+"_source"]=source
            out[phase+"_gold"]=value
            for name,m in models.items():
                inp=chain.to_mx(p)
                out[name+"_"+phase+"_soft"]=np.array(m.weights(inp).argmax(-1))
                if collect_greedy:
                    out[name+"_"+phase+"_discrete"]=discrete_prediction(m,p)
        if collect_greedy:
            out.update({"input_"+name:a for name,a in zip(chain.Records._fields,base)})
            out.update(current_link=programs[1].a_links,stale_link=programs[2].a_links)
        rows.append(out)
    return {k:np.concatenate([p[k] for p in rows]).astype(np.uint8) for k in rows[0]}


def score(a):
    table={}
    hits={}
    for arm in ARMS:
        for recipe in ("clean","write"):
            key=arm+"_"+recipe
            for q in ("soft","discrete"):
                item=key+"_"+q
                h={}
                for p in write.PHASES:
                    ids=a[key+"_"+p+"_"+q]
                    answers=np.take_along_axis(a["input_payloads"],ids,axis=1)
                    h[p]=answers==a[p+"_gold"]
                hits[item]=h
                table[item]={p:dict(accuracy=float(v.mean()),correct=int(v.sum()),queries=v.size,
                    interval=write.ci(v.mean(1))) for p,v in h.items()}
                both=h["current"]&h["stale"]
                table[item].update(both_edits=float(both.mean()),complete_edited_program=float(both.all(1).mean()))
    changes={}
    for arm in ARMS:
        anchor=hits[arm+"_clean_soft"]
        for recipe in ("clean","write"):
            key=arm+"_"+recipe+"_discrete"
            changes[key]={p:dict(delta=float(d.mean()),interval=write.ci(d)) for p,d in
                ((p,q.mean(1)-anchor[p].mean(1)) for p,q in hits[key].items())}
    flags={}
    for arm in ARMS:
        for recipe in ("clean","write"):
            k=arm+"_"+recipe+"_discrete"
            stats=table[k]
            clean=changes[k]["clean"]
            flags[k]=dict(clears_edits=min(stats[p]["accuracy"] for p in write.EDITS)>=.90,
                edit_intervals_clear=min(stats[p]["interval"][0] for p in write.EDITS)>.90,
                clean_held=clean["delta"]>=0 and clean["interval"][0]>-.01)
    return dict(scores=table,changes=changes,decision=flags)


def main():
    mx.set_default_device(mx.cpu)
    mx.set_cache_limit(64*1024*1024)
    donor=chain.source(101,"reconstruct original donor")
    witness=chain.verify_donor(donor)
    fits={}
    for seed,pretrain_seed in ((503,341),(607,861)):
        native=chain.source(pretrain_seed,f"{seed}: matched source initializer")
        for host,arch,offset in (("convolution",chain.Convolution,64),("recurrent",chain.Recurrent,128)):
            label=f"{host}-{seed}"
            models={}
            for arm,initializer in (("donor",donor),("scratch",None),("pretrained",native)):
                for recipe,run in (("clean",chain.train),("write",write.fit_write)):
                    mx.random.seed(seed)
                    m=chain.Host(arch,offset)
                    if initializer is not None:
                        m.locator.update(initializer.locator.parameters())
                    name=arm+"_"+recipe
                    run(m,42918+seed,f"{label}/{name}")
                    models[name]=m
            ref=BASE/(label+".npz")
            replay=score_panel(models,offset,968307+seed,False)
            with np.load(ref) as original:
                for p in write.PHASES:
                    np.testing.assert_array_equal(replay[p+"_gold"],original[p+"_gold"])
                    for k in models:
                        np.testing.assert_array_equal(replay[k+"_"+p+"_soft"],original[k+"_"+p+"_source"])
            fresh=score_panel(models,offset,1688701+seed,True)
            np.savez_compressed(OUT/(label+".npz"),**fresh)
            fits[label]=dict(primary=score(fresh),archive_sha256=hashlib.sha256(ref.read_bytes()).hexdigest())
    atomic_json("result.json",dict(decision="complete",fits=fits,steps=chain.TARGET_STEPS,
        donor_witness_sha256=witness,device=str(mx.default_device()),hardware=mx.device_info()["device_name"]))


if __name__=="__main__":
    main()
