"""Reopen the symmetric memory-bank test with a tabular conditional local router."""
from functools import partial
from pathlib import Path
import sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"learned_banks"))
from experiment import Banks,accuracy,evaluate,neighbors,panel,rollout,sample,sensors,mx,nn,optim
from experiment_persistence import initialize
sys.path.insert(0,str(ROOT/"reference_lm"))
from train import OUT,restore,snapshot,write
from mlx.utils import tree_flatten

class ConditionalBanks(Banks):
    def __init__(self,biased):
        super().__init__(biased)
        self.extra=mx.zeros((8*256,2,14))

    def tick(self,h,mem,a,v):
        local=sensors(v)
        pattern=(local.astype(mx.int32)*(2**mx.arange(8))).sum(-1)
        offsets=self.extra[256*a+pattern]
        weights=mx.softmax(self.table[a]+offsets[...,None,:],axis=-1)
        gate=mx.sigmoid(local@self.gate+self.bias)[...,None]
        mix=weights[...,0,:]+gate*(weights[...,1,:]-weights[...,0,:])
        sources=mx.stack([*neighbors(h,mx),*neighbors(mem,mx)],axis=-2)
        state=mix@sources
        return state[...,0,:],state[...,1,:]

def fit(name):
    mx.random.seed(101)
    model=ConditionalBanks(name=="biased")
    if name=="persistent":initialize(model)
    opt=optim.Adam(.08);path=OUT/(name+".safetensors")
    start=restore(model,opt,path) if path.exists() else 0
    def objective(m,x,a,v,y,flag):
        out,_=rollout(m,x,a,v,flag)
        return mx.logaddexp(0.,-5.*out*y).mean()
    vg=nn.value_and_grad(model,objective);state=[model.state,opt.state]
    @partial(mx.compile,inputs=state,outputs=state)
    def step(x,a,v,y,flag):
        loss,grad=vg(model,x,a,v,y,flag);opt.update(model,grad);return loss
    rng=np.random.default_rng(61904+101)
    for t in range(1,501):
        x,a,v,y,_=sample(rng,12,4,8)
        if t<=start:continue
        loss=step(*map(mx.array,(x,a,v,y,np.zeros(12,dtype=bool))));mx.eval(loss,state)
        if t%25==0:
            if not np.isfinite(loss.item()):raise ValueError("nonfinite loss")
            snapshot(model,opt,t,path);write("progress.json",dict(arm=name,step=t,target=500,loss=loss.item(),time=time.time()))
    return model

def main():
    if sys.argv[1:]==["smoke"]:
        mx.set_default_device(mx.cpu)
        for biased in (False,True):
            mx.random.seed(101);a=Banks(biased)
            mx.random.seed(101);b=ConditionalBanks(biased)
            if not biased:initialize(a);initialize(b)
            x,cmd,v,_,_=sample(np.random.default_rng(12),2,4,3)
            x,cmd,v=map(mx.array,(x,cmd,v))
            for i,j in zip(a.tick(x,-x,cmd[0],v[0]),b.tick(x,-x,cmd[0],v[0])):
                np.testing.assert_allclose(np.array(i),np.array(j),atol=1.e-6)
        print("Identical initial transitions; trainable weights",sum(v.size for _,v in tree_flatten(b.parameters())))
        return
    models={k:fit(k) for k in ("persistent","biased")}
    summary={};samples={}
    for name,m in models.items():
        valid=evaluate(m,name,panel(371239,4,8,8))
        bank=int(np.argmax([accuracy(valid["probe0"]),accuracy(valid["probe1"])]))
        held=evaluate(m,name,panel(752916,6,16,16),True)
        old=np.load(ROOT/"persistence/results"/(name+"-101.npz"))
        for k,v in held.items():
            np.testing.assert_array_equal(v[:,0],old[k][:,0]);samples[name+"_"+k]=v
        lhs,rhs=held["primary"],old["primary"]
        rng=np.random.default_rng(829731);n=len(lhs)
        # Pairs of volumes are partners in the bank-exchange intervention.
        pairs=rng.integers(n//2,size=(4000,n//2))
        ids=np.stack([2*pairs,2*pairs+1],axis=-1).reshape(4000,n)
        big,small=lhs[ids].sum(1),rhs[ids].sum(1)
        interval=np.quantile(big[:,1]/big[:,0]-small[:,1]/small[:,0],[.025,.975]).tolist()
        summary[name]=dict(validation=accuracy(valid["primary"]),read_bank=bank,large=accuracy(lhs),
             small=accuracy(rhs),improvement_interval=interval,
             selective_disruption=accuracy(held["corrupt"+str(1-bank)])-accuracy(held["corrupt"+str(bank)]))
    np.savez_compressed(OUT/"counts.npz",**samples)
    x=summary["persistent"];passed=x["large"]>=.9 and x["improvement_interval"][0]>0 and x["selective_disruption"]>=.25
    stress={}
    if passed:
        for name,m in models.items():
            z=evaluate(m,name,panel(831714,16,32,4))
            np.savez_compressed(OUT/(name+"-4096-cells.npz"),**z);stress[name]=accuracy(z["primary"])
    write("result.json",dict(decision="replicate" if passed else "stop",weights=57812,arms=summary,large_grid=stress))

if __name__=="__main__":main()
