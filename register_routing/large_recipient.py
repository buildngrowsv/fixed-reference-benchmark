"""Splice the frozen compact donor into one head of a 227M-parameter receiver."""
import sys
import numpy as np
from distill_interface import Student
from head_surgery import adapt, freeze_except_receiver
from large_revisit import WIDTH
from run import RUNS,OUT,RoutedModel,batch,mx,restore,optimizer,evaluate,scores,uncertainty,write

def host():
    m=RoutedModel(WIDTH)
    step=restore(m,optimizer(),RUNS/"large-register-dense-vs-displaced/dense.safetensors")
    assert step==512
    return m

def main():
    if sys.argv[1:]==["smoke"]:
        mx.set_default_device(mx.cpu)
        m=RoutedModel(WIDTH)
        s=Student()
        assert restore(s,optimizer(),RUNS/"distilled-memory-interface/aligned.safetensors")==4096
        x=mx.array(batch([681591,0],1)[0])
        logits=np.array(m(x,wrong=True,first_head=s(x)))
        assert logits.shape==(1,64,32) and np.isfinite(logits).all()
        print("Large receiver trainable weights",freeze_except_receiver(m))
        return
    baseline=host()
    screen=evaluate(baseline,"rotated",72718)
    validation=float(np.mean(scores(screen)))
    if validation>=.85:
        np.save(OUT/"screen.npy",screen)
        write("result.json",dict(decision="stop",reason="native validation ceiling",balanced=validation))
        return
    counts={};ledger=[];models={}
    for arm in ("native","aligned","permuted"):
        m=host()
        student=None
        if arm!="native":
            student=Student()
            assert restore(student,optimizer(),RUNS/"distilled-memory-interface"/(arm+".safetensors"))==4096
        if arm=="aligned":
            counts["before_aligned"]=evaluate(lambda x,wrong: m(x,wrong=wrong,first_head=student(x)),"rotated",518263)
        trainable=adapt(m,arm,student,741752,checkpoint_every=256)
        models[arm]=(m,student)
    models["donor_off"]=(models["aligned"][0],None)
    for i in range(32):
        ids,y,op,changed,_=batch([518263,i]);x=mx.array(ids)
        p,e=np.where(changed&(op>=2))
        row=[i*32+p,e,op[p,e],y[p,e]]
        for name,(m,source) in models.items():
            pred=np.array(m(x,wrong=True,first_head=None if source is None else source(x)).argmax(-1))
            row.append(pred[p,e])
        ledger.append(np.column_stack(row))
    q=np.concatenate(ledger);np.save(OUT/"queries.npy",q)
    for j,k in enumerate(models):
        rows=np.zeros((1024,6),dtype=int)
        for t,c in ((2,0),(3,3)):
            use=q[:,2]==t
            rows[:,c]=np.bincount(q[use,0],minlength=1024)
            rows[:,c+1]=np.bincount(q[use&(q[:,j+4]==q[:,3]),0],minlength=1024)
            rows[:,c+2]=counts["before_aligned"][:,c+2]
        np.testing.assert_array_equal(rows[:,[0,3]],counts["before_aligned"][:,[0,3]])
        counts[k]=rows
    np.savez_compressed(OUT/"counts.npz",**counts)
    gains={}
    for k in ("native","permuted","donor_off","before_aligned"):
        gains[k]=dict(delta=float(np.mean(scores(counts["aligned"]))-np.mean(scores(counts[k]))),
                      interval=uncertainty(counts["aligned"],counts[k]))
    new=(q[:,5]==q[:,3])&(q[:,4]!=q[:,3])
    supported=all(gains[k]["delta"]>=.05 and gains[k]["interval"][0]>0 for k in ("native","permuted","donor_off"))
    write("result.json",dict(decision="replicate" if supported else "stop",balanced_validation=validation,
          accuracy={k:scores(v) for k,v in counts.items()},effects=gains,trainable=trainable,
          graft_new=int(new.sum()),new_lost_off=int((new&(q[:,7]!=q[:,3])).sum()),
          native_solves_lost=int(((q[:,4]==q[:,3])&(q[:,5]!=q[:,3])).sum()),
          off_rescues=int(((q[:,5]!=q[:,3])&(q[:,7]==q[:,3])).sum())))

if __name__=="__main__":main()
