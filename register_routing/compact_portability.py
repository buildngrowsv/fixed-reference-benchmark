"""Frozen compact branch, different recipient, and a directional value exchange."""
import sys
import numpy as np
from distill_interface import Student
from head_surgery import DONOR,load_model
from prefix_surgery import source_head
from run import RUNS,OUT,RoutedModel,mx,restore,optimizer,batch,evaluate,scores,uncertainty,write

def main():
    smoke=sys.argv[1:]==["smoke"]
    if smoke:mx.set_default_device(mx.cpu)
    host=RoutedModel();assert restore(host,optimizer(),RUNS/"displaced-independent-host/trained.safetensors")==1024
    student=Student();assert restore(student,optimizer(),RUNS/"distilled-memory-interface/aligned.safetensors")==4096
    donor=load_model(DONOR)
    sources={"compact":lambda x,k:student(x,exchange=k),"full":lambda x,k:source_head(donor,donor,x,exchange=k)}
    counts={}
    if not smoke:
        for name,source in sources.items():
            def run(x,wrong):
                return host(x,wrong=True,first_head=source(x,None))
            counts[name]=evaluate(run,"rotated",272461)
    trials=[]
    for i in range(1 if smoke else 128):
        a,b=batch([35719,i],4 if smoke else 32,True)
        ids=mx.array(np.concatenate([a[0][::2],b[0][::2]]));lab=np.stack([a[1][::2,-1],b[1][::2,-1]],axis=1)
        n=len(lab);alt=mx.array(np.r_[n:2*n,0:n]);ev=a[0][::2].reshape(n,64,4)
        p=(ev[:,:16,1]==ev[:,-1:,1]).argmax(1);all_results=[lab]
        for source in sources.values():
            def score(ex):
                return np.array(host(ids,wrong=True,first_head=source(ids,ex))[:,-1])
            raw=score(None);all_results.append(raw.argmax(1).reshape(2,n).T)
            for loc in (4*(p+1)+3,np.full(n,167)):
                pos=mx.array(np.tile(loc,2))
                if i==0:np.testing.assert_array_equal(score((pos,mx.arange(2*n))),raw)
                all_results.append(score((pos,alt)).argmax(1).reshape(2,n).T)
        trials.append(np.stack(all_results,axis=1))
    if smoke:
        print("Both source self-exchanges exactly reconstruct frozen independent host")
        return
    pairs=np.concatenate(trials);gold=pairs[:,0];details={}
    for k,col in (("compact",1),("full",4)):
        clean=(pairs[:,col]==gold).all(1);a=clean&(pairs[:,col+1]==gold[:,::-1]).all(1);b=clean&(pairs[:,col+2]==gold[:,::-1]).all(1)
        d=a.astype(float)-b;indexes=np.random.default_rng(291637).integers(len(d),size=(4000,len(d)))
        details[k]=dict(clean=int(clean.sum()),source=int(a.sum()),control=int(b.sum()),delta=float(d.mean()),interval=np.quantile(d[indexes].mean(1),[.025,.975]).tolist())
    np.savez_compressed(OUT/"evidence.npz",pairs=pairs,**counts)
    ci=uncertainty(counts["compact"],counts["full"]);m=details["compact"]
    write("result.json",dict(decision="complete",accuracy={k:scores(v) for k,v in counts.items()},compact_minus_full_interval=ci,
          pairs=details,qualified=bool(ci[0]>-.05 and m["delta"]>=.25 and m["interval"][0]>0)))

if __name__=="__main__":main()
