from functools import partial
import json
from pathlib import Path
import sys
import time
from bypass import Bypass,CodeLM,Corpus,OUT,RUNS,evaluate,interval,write,restore,snapshot,tree_flatten,mx,nn,optim,np

def qualifies(params,upper):
    return bool(params<=.85*31992320 and upper<.02)

def main():
    if sys.argv[1]=="gate":
        initial=json.loads((RUNS/"sealed-bypass-main/result.json").read_text())
        upper=initial["metrics"]["val"]["fixed"]["ci"][1]
        passed=qualifies(initial["parameters"],upper)
        write("result.json",dict(decision="replicate" if passed else "stop",parameters=initial["parameters"],validation_upper=upper))
        return
    corpus=Corpus();arms={};samples={};end=16384;groups={}
    working_only = sys.argv[1] == "working-exit"
    models = (("working", partial(Bypass, working_exit=True)),) if working_only else (("sealed",Bypass),("readable",CodeLM))
    for name,modeltype in models:
        mx.random.seed(211)
        m=modeltype()
        opt=optim.AdamW(lambda t:.0004*mx.minimum(1.,(t.astype(mx.float32)+1)/128),weight_decay=.03,bias_correction=True)
        target=OUT/(name+".safetensors")
        start=restore(m,opt,target) if target.exists() else 0
        def loss(model,x,y):return nn.losses.cross_entropy(model(x,True),y,reduction="mean")
        vg=nn.value_and_grad(m,loss);state=[m.state,opt.state]
        @partial(mx.compile,inputs=state,outputs=state)
        def step(x,y):
            v,g=vg(m,x,y);g,_=optim.clip_grad_norm(g,1.);opt.update(m,g);return v
        clock=time.monotonic()
        for t in range(start+1,end+1):
            b,*_=corpus.batch("train",[41923,211,t],16)
            v=step(mx.array(b[:,:-1]),mx.array(b[:,1:]));mx.eval(v,state)
            if t%64==0:
                if not np.isfinite(v.item()):raise ValueError("nonfinite loss")
                snapshot(m,opt,t,target)
                write("progress.json",dict(arm=name,step=t,target=end,loss=v.item(),time=time.time(),
                    tokens_per_second=(t-start)*8192/(time.monotonic()-clock)))
        arms[name]=dict(parameters=sum(w.size for _,w in tree_flatten(m.parameters())),training_seconds=time.monotonic()-clock,start=start)
        for split,n in (("val",1024),("test",2048)):
            v,g,w=evaluate(m,True,corpus,split,n)
            if split in groups:
                np.testing.assert_array_equal(g,groups[split][0]);np.testing.assert_array_equal(w,groups[split][1])
            groups[split]=(g,w)
            np.savez_compressed(OUT/(name+"-"+split+".npz"),scores=v,groups=g,windows=w)
            arms[name][split+"_bpb"]=float(v[:,0].mean()/512)
            samples[(name,split)]=v
    if working_only:
        comparisons = {}
        for split, (g, w) in groups.items():
            with np.load(RUNS/"sealed-bypass-repeat"/("sealed-"+split+".npz")) as ref:
                np.testing.assert_array_equal(ref["groups"], g)
                np.testing.assert_array_equal(ref["windows"], w)
                working, sealed = samples["working",split], ref["scores"]
                comparisons[split] = dict(delta=float((working[:,0]-sealed[:,0]).mean()/512),
                                           interval=interval(working, sealed, g))
        assert arms["working"]["parameters"] == 26830464
        write("result.json",dict(decision="complete",seed=211,arms=arms,working_minus_sealed=comparisons))
        return
    ci={s:interval(samples["sealed",s],samples["readable",s],groups[s][0]) for s in groups}
    write("result.json",dict(decision="complete",seed=211,arms=arms,intervals=ci,
        qualifies=qualifies(arms["sealed"]["parameters"],ci["test"][1])))
if __name__=="__main__":main()
