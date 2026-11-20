"""Check before/after causal attribution and separately initialized host results."""
import json,sys
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root.parent/"reference_lm"))
from programs import batch

def balanced(t):
    return (t[...,1]/t[...,0]+t[...,4]/t[...,3])/2

def compare(arms, terms, reported):
    rng=np.random.default_rng(192841);draw=[]
    n=len(arms[terms[0][0]])
    delta=sum(sign*balanced(arms[key].sum(0)) for key,sign in terms)
    for _ in range(160):
        idx=rng.choice(n,(25,n),replace=True)
        draw.extend(sum(sign*balanced(arms[key][idx].sum(1)) for key,sign in terms))
    ci=np.quantile(draw,[.025,.975])
    np.testing.assert_allclose([delta,*ci],[reported["delta"],*reported["interval"]])
    return dict(delta=float(delta),interval=ci.tolist())

def get_inputs(seed,n):
    for i in range(n//32):
        _,gold,op,changed,guess=batch([seed,i])
        yield i,gold,op,changed,guess

def verify_panel(arms,seed,n):
    expected=[]
    for _,y,op,changed,guess in get_inputs(seed,n):
        tally=[]
        for typ in (2,3):
            use=changed&(op==typ)
            tally.extend([use.sum(1),((guess==y)&use).sum(1)])
        expected.extend(np.column_stack(tally))
    for v in arms.values():
        assert v.shape==(n,6) and (v[:,[1,4]]<=v[:,[0,3]]).all()
        np.testing.assert_array_equal(v[:,[0,2,3,5]],expected)

def audit():
    path=root/"results/receiver-learning";record=json.loads((path/"result.json").read_text())
    arms=dict(np.load(path/"counts.npz"));ledger=np.load(path/"queries.npy")
    verify_panel(arms,712981,4096)
    expected=[]
    for i,y,op,changed,_ in get_inputs(712981,4096):
        p,e=np.where(changed&(op>=2))
        expected.extend(np.column_stack((32*i+p,e,op[p,e],y[p,e])))
    np.testing.assert_array_equal(ledger[:,:4],expected)
    for j,name in enumerate(("before_donor","after_donor","before_host","after_host")):
        right=(ledger[:,4+j]==ledger[:,3])
        for typ,col in ((2,1),(3,4)):
            idx=(ledger[:,2]==typ)&right
            np.testing.assert_array_equal(np.bincount(ledger[idx,0],minlength=4096),arms[name][:,col])
        np.testing.assert_allclose(arms[name].sum(0)[[1,4]]/arms[name].sum(0)[[0,3]],record["accuracy"][name])
    both=(ledger[:,5]==ledger[:,3])&(ledger[:,4]!=ledger[:,3])&(ledger[:,7]!=ledger[:,3])
    assert both.sum()==record["adaptation_and_source_dependent"]
    terms=[("after_donor",1),("before_donor",-1)]
    source=compare(arms,terms,record["donor_source_gain"])
    interaction=compare(arms,terms+[("after_host",-1),("before_host",1)],record["interaction"])
    assert record["qualified"]==all(x["delta"]>=.05 and x["interval"][0]>0 for x in (source,interaction))
    summary=dict(receiver={k:float(balanced(v.sum(0))) for k,v in arms.items()},donor_source_gain=source,interaction=interaction,
                 both_required=int(both.sum()),queries=len(ledger),qualified=record["qualified"])
    path=root/"results/independent-host";record=json.loads((path/"result.json").read_text())
    arms=dict(np.load(path/"test.npz"));verify_panel(arms,436179,1024)
    scores={}
    for name,v in arms.items():
        s=v.sum(0);scores[name]=float(balanced(s))
        np.testing.assert_allclose(s[[1,4]]/s[[0,3]],record["accuracy"][name])
        np.testing.assert_allclose(scores[name],record["balanced"][name])
    comparisons={k:compare(arms,[("trained",1),(k,-1)],v) for k,v in record["comparisons"].items()}
    passed=scores["trained"]>=.75 and all(comparisons[k]["delta"]>=.1 and comparisons[k]["interval"][0]>0 for k in ("initial","source_removed"))
    assert passed==(record["decision"]=="replicate")
    summary.update(independent=scores,independent_contrasts=comparisons,independent_qualified=bool(passed))
    path=root/"results/distilled";d=json.loads((path/"result.json").read_text())
    arms=dict(np.load(path/"test.npz"));verify_panel(arms,319754,1024)
    acc={}
    for name,v in arms.items():
        t=v.sum(0);np.testing.assert_allclose(t[[1,4]]/t[[0,3]],d["accuracy"][name]);acc[name]=float(balanced(t))
    comparisons={k:compare(arms,[("aligned",1),(k,-1)],v) for k,v in d["effects"].items()}
    student=(54+256)*128+(12*128**2+2*128)+128+3*128*64
    teacher=(54+256)*384+3*(12*384**2+2*384)+384+3*384*64
    assert (student,teacher)==(d["student_weights"],d["teacher_weights_used"])
    lat={k:np.array(v) for k,v in d["batch_seconds"].items()}
    assert all(len(v)==20 and np.all(v>0) for v in lat.values())
    summary["distilled"]=dict(balanced=acc,contrasts=comparisons,weights=[student,teacher],
          fraction_saved=1-student/teacher,median_batch_ms={k:float(np.median(v)*1000) for k,v in lat.items()},
          observed_speed_ratio=float(np.median(lat["teacher"])/np.median(lat["aligned"])))
    return summary

if __name__=="__main__":print(json.dumps(audit(),indent=2))
