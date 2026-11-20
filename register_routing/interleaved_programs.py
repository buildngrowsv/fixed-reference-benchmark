"""Preserve each register's event order while interleaving independent registers."""
import sys
import numpy as np
from head_surgery import DONOR,load_model
from prefix_surgery import source_head
from run import (OUT,RUNS,RoutedModel,batch,mx,restore,optimizer,scores,uncertainty,write)

def interleave(ids,rng):
    ev=ids.reshape(len(ids),64,4);perms=[]
    for row in ev:
        keys=row[:-1,1]
        new_keys=rng.permutation(keys)
        perm=np.arange(64)
        for key in np.unique(keys):
            perm[np.flatnonzero(new_keys==key)]=np.flatnonzero(keys==key)
        assert np.array_equal(np.sort(perm),np.arange(64))
        perms.append(perm)
    order=np.stack(perms)
    return ev[np.arange(len(ids))[:,None],order].reshape(ids.shape),order

def check_labels(tokens):
    events=tokens.reshape(len(tokens),64,4)
    original=np.full((len(tokens),16),-1);current=original.copy();rows=np.arange(len(tokens));labels=[];changes=[]
    for col in events.transpose(1,0,2):
        op,key,v=col[:,0],col[:,1]-4,col[:,2]-20
        init=op==0;written=op<2
        assert np.all(original[rows[init],key[init]]==-1)
        original[rows[init],key[init]]=v[init]
        current[rows[written],key[written]]=v[written]
        assert np.all(original[rows,key]>=0)
        labels.append(np.where(op==2,original[rows,key],current[rows,key]))
        changes.append(original[rows,key]!=current[rows,key])
    return np.stack(labels,axis=1),np.stack(changes,axis=1)

def main():
    smoke=sys.argv[1:]==["smoke"]
    if smoke:mx.set_default_device(mx.cpu)
    displaced=load_model(DONOR);exact=load_model(RUNS/"register-addressing/matched.safetensors")
    host=RoutedModel();assert restore(host,optimizer(),RUNS/"displaced-prefix-surgery/trained_prefix.safetensors")==1024
    models={"displaced":lambda x:displaced(x,wrong=True),"exact":lambda x:exact(x,wrong=False),
            "hybrid":lambda x:host(x,wrong=True,first_head=source_head(displaced,displaced,x))}
    counts={};orders=[]
    for i in range(1 if smoke else 64):
        ids,y,op,changed,guess=batch([270643,i],2 if smoke else 32)
        mixed,perm=interleave(ids,np.random.default_rng([131653,i]));orders.append(perm.astype(np.uint8))
        gold,ch=check_labels(mixed)
        np.testing.assert_array_equal(gold,np.take_along_axis(y,perm,axis=1))
        np.testing.assert_array_equal(ch,np.take_along_axis(changed,perm,axis=1))
        for k,model in models.items():
            for schedule,data in (("original",ids),("interleaved",mixed)):
                pred=np.array(model(mx.array(data)).argmax(-1))
                if schedule=="interleaved":
                    pred=np.take_along_axis(pred,np.argsort(perm,axis=1),axis=1)
                row=[]
                for typ in (2,3):
                    use=changed&(op==typ)
                    row.extend([use.sum(1),((pred==y)&use).sum(1),((guess==y)&use).sum(1)])
                counts.setdefault(k+"_"+schedule,[]).append(np.column_stack(row))
    if smoke:
        print("Three frozen models, unchanged register semantics and label mapping verified")
        return
    arr={k:np.concatenate(v) for k,v in counts.items()};np.savez_compressed(OUT/"scores.npz",orders=np.concatenate(orders),**arr)
    contrasts={};acc={k:scores(v) for k,v in arr.items()}
    for k in models:
        a,b=arr[k+"_original"],arr[k+"_interleaved"]
        contrasts[k]=dict(loss=float(np.mean(scores(a))-np.mean(scores(b))),interval=uncertainty(a,b))
    diff=arr["hybrid_original"].copy()
    for col in (1,4):diff[:,col]-=arr["exact_original"][:,col]-arr["exact_interleaved"][:,col]
    interaction=dict(delta=float(np.mean(scores(diff))-np.mean(scores(arr["hybrid_interleaved"]))),
                     interval=uncertainty(diff,arr["hybrid_interleaved"]))
    outcome="schedule-sensitive" if contrasts["hybrid"]["interval"][0]>.05 else ("robust" if contrasts["hybrid"]["interval"][1]<.05 else "unresolved")
    write("result.json",dict(decision="complete",outcome=outcome,accuracy=acc,paired=contrasts,excess_hybrid_loss_over_exact=interaction))

if __name__=="__main__":main()
