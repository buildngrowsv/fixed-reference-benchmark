"""Factor recipient adaptation and the source feeding the graft on new programs."""
import numpy as np
from prefix_surgery import source_head
from head_surgery import HOST,DONOR,load_model,install_head
from run import (RUNS,OUT,RoutedModel,batch,mx,optimizer,restore,scores,uncertainty,write)


def main():
    donor=load_model(DONOR)
    before=load_model(HOST);install_head(before,donor)
    after=RoutedModel()
    assert restore(after,optimizer(),RUNS/"displaced-prefix-surgery/trained_prefix.safetensors")==1024
    rows=[];counts={}
    for i in range(128):
        ids,y,op,changed,guess=batch([712981,i])
        x=mx.array(ids);features=source_head(donor,donor,x)
        p,e=np.where(changed&(op>=2));query_preds=[]
        for label,model,head in (("before_donor",before,features),("after_donor",after,features),
                                 ("before_host",before,None),("after_host",after,None)):
            pred=np.array(model(x,wrong=True,first_head=head).argmax(-1))
            query_preds.append(pred[p,e]);tally=[]
            for typ in (2,3):
                use=changed&(op==typ)
                tally.extend([use.sum(1),((pred==y)&use).sum(1),((guess==y)&use).sum(1)])
            counts.setdefault(label,[]).append(np.column_stack(tally))
        rows.append(np.column_stack((i*32+p,e,op[p,e],y[p,e],*query_preds)))
    arrays={k:np.concatenate(v) for k,v in counts.items()}
    np.savez_compressed(OUT/"counts.npz",**arrays)
    ledger=np.concatenate(rows).astype(np.int32);np.save(OUT/"queries.npy",ledger)
    # This contrast has the same per-program denominators as a single arm:
    # subtract the host-source improvement from the adapted donor-source numerator.
    interaction=arrays["after_donor"].copy()
    for col in (1,4):
        interaction[:,col]-=arrays["after_host"][:,col]-arrays["before_host"][:,col]
    def paired(a,b):
        return dict(delta=float(np.mean(scores(a))-np.mean(scores(b))),interval=uncertainty(a,b))
    ordinary=paired(arrays["after_donor"],arrays["before_donor"])
    effect=paired(interaction,arrays["before_donor"])
    successes=(ledger[:,5]==ledger[:,3]) & (ledger[:,4]!=ledger[:,3]) & (ledger[:,7]!=ledger[:,3])
    write("result.json",dict(decision="complete",programs=4096,accuracy={k:scores(v) for k,v in arrays.items()},
          donor_source_gain=ordinary,interaction=effect,adaptation_and_source_dependent=int(successes.sum()),
          qualified=bool(effect["delta"]>=.05 and effect["interval"][0]>0 and ordinary["delta"]>=.05 and ordinary["interval"][0]>0)))


if __name__=="__main__":main()
