"""Fit receivers in an archived weak host that never shared the donor initialization."""
import sys
import numpy as np
from head_surgery import DONOR,adapt,install_head,load_model
from prefix_surgery import source_head
from run import RUNS,OUT,RoutedModel,batch,evaluate,mx,scores,uncertainty,write
HOST=RUNS/"register-stable-random-second-seed/random.safetensors"
PANEL=436179

def main():
    donor=load_model(DONOR);host=load_model(HOST)
    assert not np.array_equal(np.array(host.tokens.weight),np.array(donor.tokens.weight))
    mx.random.seed(101)
    initial=RoutedModel()
    def scored(m,prefix):
        def live(x,wrong):
            return m(x,wrong=wrong,first_head=source_head(prefix,donor,x))
        return evaluate(live,"rotated",PANEL)
    if sys.argv[1:]==["smoke"]:
        mx.set_default_device(mx.cpu)
        install_head(host,donor)
        x=mx.array(batch([PANEL,1],2)[0])
        z=np.array(host(x,wrong=True,first_head=source_head(donor,donor,x)))
        assert z.shape==(2,64,32) and np.isfinite(z).all()
        print("Distinct host checkpoint and joint inference verified")
        return
    results={"donor":evaluate(donor,"rotated",PANEL),"native_on_displaced_mask":evaluate(host,"rotated",PANEL)}
    for arm,prefix in (("trained",donor),("initial",initial)):
        m=load_model(HOST)
        install_head(m,donor)
        if arm=="trained":results["before"]=scored(m,prefix)
        def source(x):
            return source_head(prefix,donor,x)
        adapt(m,arm,source,training_seed=741752)
        results[arm]=scored(m,prefix)
        if arm=="trained":results["source_removed"]=evaluate(m,"rotated",PANEL)
    np.savez_compressed(OUT/"test.npz",**results)
    accuracy={k:scores(v) for k,v in results.items()}
    diff={}
    for k in ("initial","source_removed","before"):
        np.testing.assert_array_equal(results[k][:,[0,2,3,5]],results["trained"][:,[0,2,3,5]])
        diff[k]=dict(delta=float(np.mean(accuracy["trained"])-np.mean(accuracy[k])),
                     interval=uncertainty(results["trained"],results[k]))
    passes=np.mean(accuracy["trained"])>=.75 and all(diff[k]["delta"]>=.10 and diff[k]["interval"][0]>0 for k in ("initial","source_removed"))
    write("result.json",dict(decision="replicate" if passes else "stop",balanced={k:float(np.mean(v)) for k,v in accuracy.items()},
          accuracy=accuracy,comparisons=diff,programs=len(results["trained"])))

if __name__=="__main__":main()
