"""Scale the original readable fixed-versus-writable comparison at matched data exposure."""
import argparse
from functools import partial
import time
import numpy as np
from train_code import CodeLM,Corpus,OUT,RUNS,mx,nn,optim,tree_flatten,restore,snapshot,evaluate,write
from verify import grouped_difference

def fit(arm,corpus,phase,end,make_model=None,peak_rate=.0004,mid_run="large-python-reference-mid"):
    mx.random.seed(101)
    model=CodeLM(width=1536,protected_width=192) if make_model is None else make_model()
    count=sum(v.size for _,v in tree_flatten(model.parameters()))
    opt=optim.AdamW(lambda t:peak_rate*mx.minimum(1.,(t.astype(mx.float32)+1)/128),weight_decay=.03,bias_correction=True)
    path=OUT/(arm+".safetensors")
    start=0
    if path.exists():start=restore(model,opt,path)
    elif phase=="main":
        start=restore(model,opt,RUNS/mid_run/path.name)
        assert start==4096
    fixed=arm=="fixed"
    def objective(m,x,y):return nn.losses.cross_entropy(m(x,fixed),y,reduction="mean")
    grad=nn.value_and_grad(model,objective);state=[model.state,opt.state]
    @partial(mx.compile,inputs=state,outputs=state)
    def step(x,y):
        v,g=grad(model,x,y);g,_=optim.clip_grad_norm(g,1.);opt.update(model,g);return v
    clock=time.monotonic()
    for t in range(start+1,end+1):
        data,*_=corpus.batch("train",[41923,101,t],16)
        loss=step(mx.array(data[:,:-1]),mx.array(data[:,1:]));mx.eval(loss,state)
        if t==1 or t%16==0:
            if not np.isfinite(loss.item()):raise ValueError("nonfinite training loss")
            if t%256==0:snapshot(model,opt,t,path)
            write("progress.json",dict(arm=arm,step=t,target=end,parameters=count,loss=loss.item(),time=time.time(),
                  tokens_per_second=(t-start)*8192/(time.monotonic()-clock)))
    return model,count

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=("mid","main"))
    parser.add_argument("--width-rate",action="store_true")
    args=parser.parse_args()
    phase=args.phase
    end={"mid":4096,"main":16384}[phase]
    rate=.0004/3 if args.width_rate else .0004
    mid_run="large-python-width-rate-mid" if args.width_rate else "large-python-reference-mid"
    corpus=Corpus();arms={};arrays={}
    panels=[("validation","val",1024)]
    if phase=="main":panels.append(("test","test",2048))
    for arm in ("fixed","writable"):
        model,count=fit(arm,corpus,phase,end,peak_rate=rate,mid_run=mid_run)
        arms[arm]={"parameters":count}
        for label,split,n in panels:
            v,g,w=evaluate(model,arm=="fixed",corpus,split,n)
            np.savez_compressed(OUT/(arm+"-"+label+".npz"),scores=v,groups=g,windows=w)
            arms[arm][label+"_bpb"]=float(v[:,0].mean()/512)
            arrays[arm,label]=v[:,0]
            with np.load(RUNS/("cpython-reference-"+phase)/(arm+"-"+label+".npz")) as ref:
                np.testing.assert_array_equal(w,ref["windows"]);np.testing.assert_array_equal(g,ref["groups"])
                arrays["small_"+arm,label]=ref["scores"][:,0]
            arrays["groups",label]=g
            if args.width_rate and phase=="mid":
                with np.load(RUNS/"large-python-reference-mid"/(arm+"-"+label+".npz")) as original:
                    np.testing.assert_array_equal(w,original["windows"])
                    np.testing.assert_array_equal(g,original["groups"])
                    arms[arm]["lower_minus_original_rate_bpb"]=grouped_difference(v[:,0],original["scores"][:,0],g)
        del model
    results={}
    for label,_,_ in panels:
        f,w=arrays["fixed",label],arrays["writable",label]
        sf,sw=arrays["small_fixed",label],arrays["small_writable",label];g=arrays["groups",label]
        results[label]=dict(small=grouped_difference(sf,sw,g),large=grouped_difference(f,w,g),
                            size_interaction=grouped_difference(f-w,sf-sw,g))
    outcome="complete"
    if phase=="mid":outcome="advance" if max(v["validation_bpb"] for v in arms.values())<3.24 else "stop"
    write("result.json",dict(decision=outcome,updates=end,peak_rate=rate,arms=arms,contrasts=results,
          rates_matched_across_sizes=not args.width_rate))

if __name__=="__main__":main()
