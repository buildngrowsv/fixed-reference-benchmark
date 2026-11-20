"""Replace one in eight source-project batches with original CPython training bytes."""
import numpy as np
from ecosystem import Bypass,Corpus,Projects,RUNS,OUT,mx,train,evaluate,evaluate_projects,difference,write
from verify import grouped_difference

def main():
    base=RUNS/"scientific-python-transfer";pre=RUNS/"sealed-bypass-main"
    python=Corpus();projects=Projects(python)
    model=Bypass()
    data=mx.load(str(pre/"bypass.safetensors"));assert int(data["step"].item())==16384
    model.load_weights([(k[8:],v) for k,v in data.items() if k.startswith("weights.")]);del data
    train(model,"sealed-rehearsal",projects,rehearsal=python)
    score=evaluate_projects(model,projects);old=np.load(base/"projects.npz")
    # Compare improvement with replay against 75% of the ordinary adaptation gain.
    reference=old["sealed_before"].copy()
    reference[:,3]=.25*old["sealed_before"][:,3]+.75*old["sealed_after"][:,3]
    ext=difference(score,reference);direct=difference(score,old["sealed_after"])
    np.savez_compressed(OUT/"external.npz",scores=score)
    v,g,w=evaluate(model,True,python,"test",2048)
    prior=np.load(pre/"test.npz");adapted=np.load(base/"sealed-retention.npz")
    for f in (prior,adapted):
        np.testing.assert_array_equal(g,f["groups"]);np.testing.assert_array_equal(w,f["windows"])
    mid=(prior["scores"][:,0]+adapted["scores"][:,0])/2
    old_domain=grouped_difference(v[:,0],mid,g)
    np.savez_compressed(OUT/"cpython.npz",scores=v,groups=g,windows=w)
    good=ext["interval"][1]<0 and old_domain["interval"][1]<0
    write("result.json",dict(decision="complete",supported=bool(good),external_bpb=float(score[:,3].mean()),
          cpython_bpb=float(v[:,0].mean()/512),external_vs_75pct_gain=ext,external_vs_no_replay=direct,cpython_vs_half_penalty=old_domain))

if __name__=="__main__":main()
