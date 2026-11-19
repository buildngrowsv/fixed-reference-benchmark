"""Independent backward path tracing, weight replay and a finite-depth gate certificate."""
from itertools import product
import json
import math
import numpy as np
import mlx.core as mx
from experiment import R,ARMS,COLUMNS,Gate,evaluate
from transport import boards

def backward(idx,s):
    # Starting from every final cell, walk back to its unique original word.
    origin=np.arange(idx.shape[1])
    for row in idx[::-1]:
        there=row[origin]
        barrier=s[origin,0]^s[there,0]
        blocked=barrier & ((1-s[origin,1]) | (1-s[there,1]))
        origin=np.where(blocked.astype(bool),origin,there)
    return origin

def certificate(net,arm):
    bits=np.array(list(product((0,1),repeat=6)),np.float64)
    expected=(bits[:,0]==bits[:,3]) | ((bits[:,1]==1)&(bits[:,4]==1))
    f=bits.copy()
    if arm=="one_end": f[:,4]=0
    hidden=np.maximum(f@np.array(net.hidden.weight).astype(float).T+np.array(net.hidden.bias),0)
    logits=(hidden@np.array(net.output.weight).astype(float).T+np.array(net.output.bias))[:,0]
    p=1/(1+np.exp(-logits))
    k=expected.astype(bool)
    sign_margin=np.where(k,logits,-logits)
    # sigmoid(m) is the weight given to the truth-table branch, including when it says stay.
    # Use log-sigmoid to keep the minimum and integer depth stable.
    log_q=-np.logaddexp(0,-sign_margin.min())
    depth=int(math.ceil(math.log(.5)/log_q)-1)
    return dict(correct_table_rows=int(((p>.5)==k).sum()),table_rows=len(k),min_correct_branch_mass=float(np.exp(log_q)),
                certified_transitions=max(0,depth),row_mass=np.where(k,p,1-p).tolist())

def main():
    metadata=json.loads((R/"training.json").read_text())
    seeds=sorted(set(r["seed"] for r in metadata["training"]))
    cert={};scores={k:{} for k in seeds}
    independent_checks={}
    for val in (True,False):
        side,t,count,axes=(4,8,64,(0,)) if val else (8,16,128,(1,2))
        rng=np.random.default_rng(58619 if val else 893721)
        n_actually_changed=0
        independent=[]
        for _ in range(count//8):
            x,idx,signals,edited=boards(rng,8,side,t,axes)
            closed=signals.copy();closed[edited,1]=0
            nuisance=signals.copy();nuisance[edited,2]^=1
            a,b,c=tuple(backward(idx,s) for s in (signals,closed,nuisance))
            np.testing.assert_array_equal(a,c)
            changes=(a!=b)
            n_actually_changed+=int(np.sum(changes & np.any(x[a]!=x[b],axis=1)))
            independent.extend(changes.reshape(8,-1).sum(1))
        assert n_actually_changed==sum(independent)
        kind="val" if val else "test"
        independent_checks[kind]=dict(pairs=count,traced_dependent_sites=int(sum(independent)),distinct_words=n_actually_changed)
        for seed in seeds:
            models={}
            for arm in ARMS:
                model=Gate();model.load_weights(str(R/f"{arm}-{seed}.safetensors"));models[arm]=model
                if val:cert[f"{arm}_{seed}"]=certificate(model,arm)
            replay=evaluate(models,val)
            with np.load(R/f"{kind}-{seed}.npz") as saved:
                for arm,values in replay.items():
                    np.testing.assert_array_equal(saved[arm],values)
                    np.testing.assert_array_equal(saved[arm][:,0],independent)
                scores[seed][kind]={arm:dict(zip(COLUMNS,row.sum(0).tolist())) for arm,row in replay.items()}
    # Preserve matched boards, interventions and seeds. This contrast happens to be at the ceiling.
    rng=np.random.default_rng(83157)
    n=[];diff=[]
    for s in seeds:
        with np.load(R/f"test-{s}.npz") as z:
            n.append(z["both_ends"][:,0]);diff.append(z["both_ends"][:,1]-z["one_end"][:,1])
    i=rng.integers(count,size=(5000,1,count));j=rng.integers(len(seeds),size=(5000,len(seeds),1))
    draws=np.array(diff)[j,i].sum((1,2))/np.array(n)[j,i].sum((1,2))
    (R/"audit.json").write_text(json.dumps(dict(all_saved_counts_replayed=True,backward_trace=independent_checks,
            truth_tables=cert,paired_full_minus_blind_95_interval=np.quantile(draws,[.025,.975]).tolist(),scores=scores),indent=2)+"\n")
    print({k:(v["correct_table_rows"],v["min_correct_branch_mass"],v["certified_transitions"]) for k,v in cert.items()})
    print(independent_checks)
if __name__=="__main__":main()

