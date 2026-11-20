"""Test whether the hybrid can be made to report a specific partner's remembered value."""
from pathlib import Path
import sys
import numpy as np
from prefix_surgery import source_head
from head_surgery import DONOR,load_model
from run import OUT,RUNS,RoutedModel,batch,mx,restore,optimizer,write

def test_batch(host,donor,index,check):
    a,b=batch([192640,index],32,True)
    x=mx.array(np.concatenate([a[0][::2],b[0][::2]]))
    labels=np.stack([a[1][::2,-1],b[1][::2,-1]],axis=1)
    n=len(labels);partner=mx.array(np.r_[n:2*n,0:n])
    events=a[0][::2].reshape(n,64,4)
    start=(events[:,:16,1]==events[:,-1:,1]).argmax(1)
    locations=[(start+1)*4+3,np.full(n,41*4+3)]
    def logits(exchange):
        head=source_head(donor,donor,x,exchange)
        return np.array(host(x,wrong=True,first_head=head)[:,-1])
    clean=logits(None)
    arrays=[labels,clean.argmax(1).reshape(2,n).T]
    for loc in locations:
        slots=mx.array(np.tile(loc,2))
        if check:
            np.testing.assert_array_equal(logits((slots,mx.arange(2*n))),clean)
        arrays.append(logits((slots,partner)).argmax(1).reshape(2,n).T)
    return np.stack(arrays,axis=1)

def main():
    smoke=sys.argv[1:]==["smoke"]
    if smoke:mx.set_default_device(mx.cpu)
    donor=load_model(DONOR)
    if smoke:
        assert test_batch(donor,donor,0,True).shape==(16,4,2)
        print("Both self-interventions reproduced intact logits")
        return
    host=RoutedModel()
    assert restore(host,optimizer(),RUNS/"displaced-prefix-surgery/trained_prefix.safetensors")==1024
    arr=np.concatenate([test_batch(host,donor,i,i==0) for i in range(128)])
    np.save(OUT/"hybrid.npy",arr)
    gold,pred,init,write_ctrl=arr.transpose(1,0,2)
    clean=(gold==pred).all(1)
    a=clean&(init==gold[:,::-1]).all(1)
    b=clean&(write_ctrl==gold[:,::-1]).all(1)
    d=a.astype(float)-b
    rng=np.random.default_rng(118231)
    draws=np.concatenate([d[rng.integers(len(d),size=(100,len(d)))].mean(1) for _ in range(50)])
    ci=np.quantile(draws,[.025,.975]).tolist()
    write("result.json",dict(decision="complete",pairs=len(gold),clean=int(clean.sum()),source=int(a.sum()),control=int(b.sum()),
          delta=float(d.mean()),interval=ci,supported=bool(d.mean()>=.25 and ci[0]>0)))

if __name__=="__main__":
    main()
