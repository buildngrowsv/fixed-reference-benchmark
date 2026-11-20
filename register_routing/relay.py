"""Single-value-vector interchange at a learned, displaced register read."""
import numpy as np
from run import RoutedModel, OUT, RUNS, batch, mx, optimizer, restore, routing, write

def main():
    report={}
    for name,shift in (("rotated",1),("matched",0)):
        model=RoutedModel()
        restore(model,optimizer(),RUNS/"register-addressing"/(name+".safetensors"))
        rows=[]
        for i in range(64):
            a,b=batch([152843,i],32,True)
            # Even generator rows request the original register value.
            first,second=a[0][::2],b[0][::2]
            assert (a[2][::2,-1]==2).all()
            ids=np.concatenate([first,second]);x=mx.array(ids);n=len(first)
            ev=first.reshape(n,64,4)
            start=(ev[:,:16,1]==ev[:,-1:,1]).argmax(1)
            assert np.all(ev[np.arange(n),start,2]!=second.reshape(n,64,4)[np.arange(n),start,2])
            assert np.all(ev[:,40,0]==1) and np.array_equal(ev[:,40,1],ev[:,-1,1])
            gold=np.stack([a[1][::2,-1],b[1][::2,-1]],axis=1)
            assert (gold[:,0]!=gold[:,1]).all()
            perm=mx.array(np.r_[n:2*n,0:n]);self_perm=mx.arange(2*n)
            positions={"init":(start+shift)*4+3,"write":np.full(n,(40+shift)*4+3)}
            mask=routing(x,bool(shift))
            clean=np.array(model(x,wrong=bool(shift),mask=mask)[:,-1])
            pred=[clean.argmax(1).reshape(2,n).T]
            for kind,p in positions.items():
                slots=mx.array(np.tile(p,2))
                assert np.array(mask[mx.arange(2*n),0,-1,slots]).all()
                if i==0:
                    same=model(x,wrong=bool(shift),mask=mask,exchange=(slots,self_perm))
                    np.testing.assert_array_equal(clean,np.array(same[:,-1]))
                logits=model(x,wrong=bool(shift),mask=mask,exchange=(slots,perm))
                pred.append(np.array(logits[:,-1].argmax(-1)).reshape(2,n).T)
            rows.append(np.stack([gold,*pred],axis=1))
        arr=np.concatenate(rows);np.save(OUT/(name+".npy"),arr)
        gold=arr[:,0];clean=(arr[:,1]==gold).all(1)
        source=clean&(arr[:,2]==gold[:,::-1]).all(1)
        ctrl=clean&(arr[:,3]==gold[:,::-1]).all(1)
        d=source.astype(float)-ctrl
        draws=np.random.default_rng(90561).integers(len(d),size=(5000,len(d)))
        ci=np.quantile(d[draws].mean(1),[.025,.975]).tolist()
        report[name]=dict(pairs=len(clean),clean=int(clean.sum()),double_switch_init=int(source.sum()),
            double_switch_write=int(ctrl.sum()),delta=float(d.mean()),interval=ci,
            correct_after_init=int((arr[:,2]==gold).all(1).sum()),correct_after_write=int((arr[:,3]==gold).all(1).sum()))
    r=report["rotated"]
    write("result.json",dict(decision="complete",supported=bool(r["delta"]>=.25 and r["interval"][0]>0),arms=report))
if __name__=="__main__":main()
