"""Can the original inputs to the donor head rescue transfer into a weak host?"""
import sys
import numpy as np
from head_surgery import (DONOR, HOST, adapt, install_head, load_model)
from run import (OUT,RUNS,RoutedModel,batch,evaluate,mx,routing,scores,uncertainty,write)

def source_head(prefix, donor, ids):
    x=(prefix.tokens(ids)+prefix.pos(mx.arange(ids.shape[1]))).astype(mx.bfloat16)
    for block in prefix.blocks[:3]:
        x=block(x,False)
    b=donor.blocks[3]
    q,k,v=mx.split(b.qkv(b.norm1(x).astype(x.dtype)),3,axis=-1)
    def first(z):
        return z[:,:,:64][:,None,:,:]
    return mx.fast.scaled_dot_product_attention(first(q),first(k),first(v),scale=.125,mask=routing(ids,True))


def main():
    if sys.argv[1:]==["smoke"]:
        mx.set_default_device(mx.cpu)
        m=load_model(DONOR);x=mx.array(batch([99137,1],2)[0])
        np.testing.assert_array_equal(np.array(m(x,wrong=True)),np.array(m(x,wrong=True,first_head=source_head(m,m,x))))
        print("Reconstructed native donor logits exactly")
        return
    donor=load_model(DONOR)
    mx.random.seed(101)
    initial=RoutedModel()
    previous=np.load(RUNS/"displaced-head-surgery/test.npz")
    counts={k:previous[k] for k in ("native","graft","donor_full")}
    for name,prefix in (("trained_prefix",donor),("initial_prefix",initial)):
        model=load_model(HOST)
        install_head(model,donor)
        def source(ids):
            return source_head(prefix,donor,ids)
        trainable=adapt(model,name,source)
        def live(ids,wrong):
            return model(ids,wrong=wrong,first_head=source(ids))
        counts[name]=evaluate(live,"rotated",482731)
        if name=="trained_prefix":
            counts["host_input_restored"]=evaluate(model,"rotated",482731)
    np.savez_compressed(OUT/"test.npz",**counts)
    accuracy={k:scores(v) for k,v in counts.items()}
    effects={}
    for k in ("graft","initial_prefix","host_input_restored"):
        np.testing.assert_array_equal(counts[k][:,[0,2,3,5]],counts["trained_prefix"][:,[0,2,3,5]])
        delta=float(np.mean(accuracy["trained_prefix"])-np.mean(accuracy[k]))
        effects[k]=dict(delta=delta,interval=uncertainty(counts["trained_prefix"],counts[k]))
    gate=all(c["delta"] >= .10 and c["interval"][0]>0 for c in effects.values())
    write("result.json",dict(decision="replicate" if gate else "stop",effects=effects,accuracy=accuracy,trainable=trainable))

if __name__=="__main__":
    main()
