"""Change only trainability of the original donor in the two chain-transfer hosts."""
from pathlib import Path
import hashlib
import time
from mlx.utils import tree_flatten

import chain_transfer as chain
from learned_address import atomic_json, ci, mx, np, OUT

BASE = Path(__file__).resolve().parent / "chain_study/results"
MARGIN = .01


def route_arrays(model):
    return {key:np.array(v) for key,v in tree_flatten(model.locator.parameters())}


def fingerprint(arrays):
    sha=hashlib.sha256()
    for key,array in sorted(arrays.items()):
        sha.update(key.encode("utf-8"))
        sha.update(array.tobytes())
    return sha.hexdigest()


def exact(a,b):
    assert a.keys()==b.keys()
    for key in a:
        np.testing.assert_array_equal(a[key],b[key])


def train(donor, arch, offset, seed, frozen):
    mx.random.seed(seed)
    host=chain.Host(arch,offset)
    host.locator.update(donor.parameters())
    if frozen:
        host.locator.freeze()
    start=route_arrays(host)
    assert bool(tree_flatten(host.locator.trainable_parameters()))!=frozen
    elapsed=chain.train(host,42918+seed,f"{arch.__name__}/{seed}/"+("frozen" if frozen else "adapted"))
    finish=route_arrays(host)
    changed=sum(np.count_nonzero(start[k]!=finish[k]) for k in start)
    assert (changed==0 if frozen else changed>0)
    return host,start,dict(seconds=elapsed,changed_router_weights=int(changed),
        initial_router_sha256=fingerprint(start),final_router_sha256=fingerprint(finish))


def run_panel(host, offset, seed, size):
    panel_seed=(83639 if size==20 else 836390)+seed
    return chain.evaluate({"graft":host,"rolled":host},offset,panel_seed,size,16 if size==20 else 4)


def compare(candidate, reference, gold):
    d=(candidate==gold).mean(1)-(reference==gold).mean(1)
    return dict(delta=float(d.mean()),interval=ci(d))


def study(adapted, frozen, baseline):
    gold=baseline["gold"]
    for key,value in frozen.items():
        np.testing.assert_array_equal(value,baseline[key])
        if key.startswith("input_") or key in ("gold","address","edited_link","stale_link","current_gold","stale_gold"):
            np.testing.assert_array_equal(adapted[key],value)
    answer=adapted["graft_answer"]
    candidates={"adapted":answer,"frozen":frozen["graft_answer"],
                "pretrained_native":baseline["pretrained_answer"],"shifted":adapted["rolled_answer"]}
    accuracy={k:float((v==gold).mean()) for k,v in candidates.items()}
    contrasts={k:compare(answer,v,gold) for k,v in candidates.items() if k!="adapted"}
    sources={k:float((v==baseline["address"]).mean()) for k,v in
        (("adapted",adapted["graft_address"]),("frozen",frozen["graft_address"]),
         ("pretrained_native",baseline["pretrained_address"]))}
    causal={}
    for arm,values in (("adapted",adapted),("frozen",frozen)):
        correct=values["graft_answer"][:,:2]==gold[:,:2]
        causal[arm]={}
        for edit in ("current","stale"):
            hit=values[f"{edit}_answer"]==baseline[f"{edit}_gold"]
            causal[arm][edit]=dict(clean_correct=int(correct.sum()),both_correct=int((hit&correct).sum()),
                correct_unconditional=int(hit.sum()),all_queries=hit.size)
    recovered=(accuracy["adapted"]-accuracy["frozen"])/(accuracy["pretrained_native"]-accuracy["frozen"])
    return dict(accuracy=accuracy,source_accuracy=sources,contrasts=contrasts,causal=causal,
                fraction_point_gap_recovered=recovered,passes_margin=bool(contrasts["pretrained_native"]["interval"][0]>-MARGIN))


def main():
    mx.set_default_device(mx.cpu)
    mx.set_cache_limit(64*1024*1024)
    donated=chain.source(101,"reconstruct original donor")
    witness=chain.verify_donor(donated)
    donor=donated.locator
    start_donor={key:np.array(v).copy() for key,v in tree_flatten(donor.parameters())}
    fits={}
    for seed in (503,607):
        for name,encoder,offset in (("convolution",chain.Convolution,64),("recurrent",chain.Recurrent,128)):
            label=f"{name}-{seed}"
            adapted,s0,a=train(donor,encoder,offset,seed,False)
            frozen,s1,f=train(donor,encoder,offset,seed,True)
            exact(start_donor,s0)
            exact(s0,s1)
            with np.load(BASE/(label+".npz")) as archive:
                new=run_panel(adapted,offset,seed,20)
                old=run_panel(frozen,offset,seed,20)
                result=study(new,old,archive)
                result["baseline_sha256"]=hashlib.sha256((BASE/(label+".npz")).read_bytes()).hexdigest()
            a_familiar=run_panel(adapted,offset,seed,12)
            f_familiar=run_panel(frozen,offset,seed,12)
            np.testing.assert_array_equal(a_familiar["gold"],f_familiar["gold"])
            result["familiar"]=dict(adapted=float((a_familiar["graft_answer"]==a_familiar["gold"]).mean()),
                frozen=float((f_familiar["graft_answer"]==f_familiar["gold"]).mean()))
            result["training"]=dict(adapted=a,frozen=f)
            fits[label]=result
            # Retain the new predictions only. Identical inputs and archived controls have one canonical copy.
            np.savez_compressed(OUT/(label+".npz"),**{k:v for k,v in new.items()
                if k.endswith("_answer") or k.endswith("_address")})
    exact(start_donor,{k:np.array(v) for k,v in tree_flatten(donor.parameters())})
    atomic_json("result.json",dict(decision="complete",fits=fits,margin=MARGIN,
        closes_gap=all(f["passes_margin"] for f in fits.values()),exact_frozen_replay=True,
        identical_initial_router_sha256=fingerprint(start_donor),donor_witness_sha256=witness,
        target_steps=chain.TARGET_STEPS,device=str(mx.default_device()),hardware=mx.device_info()["device_name"]))


if __name__=="__main__":
    main()
