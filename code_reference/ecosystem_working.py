"""Is the early copy necessary for the compact model on external Python projects?"""
import json
import numpy as np
from ecosystem import Bypass,Corpus,OUT,RUNS,Projects,difference,evaluate_projects,mx,train,write,evaluate

def main():
    parent=RUNS/"scientific-python-transfer"
    prior=json.loads((parent/"result.json").read_text())
    if not prior["within_margin"]:
        write("result.json",dict(decision="stop",reason="sealed transfer outside its registered margin"))
        return
    python=Corpus();projects=Projects(python);base=json.loads((parent/"files.json").read_text())
    assert base["files"]==projects.catalog and base["projects"]==projects.splits
    model=Bypass(working_exit=True);path=RUNS/"working-exit-main/bypass.safetensors"
    weights=mx.load(str(path));assert int(weights["step"].item())==16384
    model.load_weights([(k[8:],v) for k,v in weights.items() if k.startswith("weights.")]);del weights
    before=evaluate_projects(model,projects)
    train(model,"working",projects)
    after=evaluate_projects(model,projects)
    refs=np.load(parent/"projects.npz")
    cmp={s:difference(v,refs["sealed_"+s]) for s,v in (("before",before),("after",after))}
    np.savez_compressed(OUT/"projects.npz",before=before,after=after)
    v,g,w=evaluate(model,True,python,"test",2048)
    np.savez_compressed(OUT/"working-retention.npz",scores=v,groups=g,windows=w)
    with np.load(RUNS/"working-exit-main/test.npz") as old:
        np.testing.assert_array_equal(w,old["windows"]);np.testing.assert_array_equal(g,old["groups"])
        retention=float((v[:,0]-old["scores"][:,0]).mean()/512)
    out=cmp["after"]
    within=out["interval"][1]<.02 and sum(d<.02 for d in out["project_differences"])>=5
    write("result.json",dict(decision="complete",working_minus_sealed=cmp,within_margin=bool(within),
          bpb={"before":float(before[:,3].mean()),"after":float(after[:,3].mean())},cpython_loss_change=retention))

if __name__=="__main__":main()
