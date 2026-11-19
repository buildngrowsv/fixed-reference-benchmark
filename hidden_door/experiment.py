"""End-to-end local gates with matched remote-latch blinding."""
from functools import partial
from pathlib import Path
import json,time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from transport import boards,readings,trace

R=Path(__file__).resolve().parent/"results"
COLUMNS=("dependent","both","open","sealed","irrelevant_open","changed_relevant","changed_irrelevant",
         "all_cells","all_open","all_sealed","both_irrelevant")
ARMS=("both_ends","one_end")

class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden=nn.Linear(6,16)
        self.output=nn.Linear(16,1)
    def __call__(self,f):
        return mx.sigmoid(self.output(nn.relu(self.hidden(f))))

def rollout(net,words,index,features):
    gates=net(features)
    for step in range(len(index)):
        words=words+gates[step]*(words[index[step]]-words)
    return words

def features(index,signals,arm):
    f=readings(index,signals).astype(np.float32)
    if arm=="one_end": f[...,4]=0
    return mx.array(f)

def save(name,record):
    (R/name).write_text(json.dumps(record,indent=2)+"\n")

def train(seed,arm):
    rng=np.random.default_rng(16273+seed)
    mx.random.seed(seed)
    net=Gate()
    opt=optim.Adam(.015)
    def loss(net,words,index,f,target):
        return mx.logaddexp(0.,-5*rollout(net,words,index,f)*target).mean()
    vg=nn.value_and_grad(net,loss)
    state=[net.state,opt.state]
    @partial(mx.compile,inputs=state,outputs=state)
    def step(words,index,f,target):
        v,g=vg(net,words,index,f,target)
        opt.update(net,g)
        return v
    hist=[]
    start=time.perf_counter()
    for t in range(1,501):
        x,idx,s,edited=boards(rng,10,4,8)
        s[edited,1]=rng.integers(2,size=10)
        goal=x[trace(idx,s)]
        v=step(mx.array(x),mx.array(idx),features(idx,s,arm),mx.array(goal))
        mx.eval(v,state)
        if t in (1,25,100,250,500): hist.append([t,v.item()])
    net.save_weights(str(R/f"{arm}-{seed}.safetensors"))
    return net,dict(seed=seed,arm=arm,updates=500,seconds=time.perf_counter()-start,history=hist,device=mx.device_info()["device_name"])

def endpoints(guess,needed,targets,batch):
    a,b,n=guess
    oa,ob=targets
    eo=(a==oa).all(-1)
    es=(b==ob).all(-1)
    en=(n==oa).all(-1)
    changed=(a!=b).any(-1);noise=(a!=n).any(-1)
    metrics=(needed,needed&eo&es,needed&eo,needed&es,needed&en,needed&changed,needed&noise,
             np.ones_like(needed),eo,es,needed&en&es)
    return np.stack([r.reshape(batch,-1).sum(1) for r in metrics],axis=1)

def evaluate(models,validation):
    rng=np.random.default_rng(58619 if validation else 893721)
    side,t,count,axes=(4,8,64,(0,)) if validation else (8,16,128,(1,2))
    metrics={}
    for _ in range(count//8):
        x,idx,s,edited=boards(rng,8,side,t,axes)
        closed=s.copy();closed[edited,1]=0
        noise=s.copy();noise[edited,2]^=1
        oa,ob=trace(idx,s),trace(idx,closed)
        needed=oa!=ob
        ref=[x[oa]>0,x[ob]>0]
        guesses={"interpreter":(ref[0],ref[1],ref[0]),"unrestricted":(x[trace(idx,s,True)]>0,)*3}
        for arm,net in models.items():
            guesses[arm]=tuple(np.array(rollout(net,mx.array(x),mx.array(idx),features(idx,signals,arm)))>0
                               for signals in (s,closed,noise))
        for name,pred in guesses.items():
            metrics.setdefault(name,[]).append(endpoints(pred,needed,ref,8))
    return {k:np.concatenate(v) for k,v in metrics.items()}

def main():
    R.mkdir(exist_ok=True)
    metadata=[];decision=None
    for seed in (101,211,307):
        models={}
        for arm in ARMS:
            net,record=train(seed,arm)
            models[arm]=net;metadata.append(record)
        val=evaluate(models,True)
        totals={k:v.sum(0) for k,v in val.items()}
        full,blind=totals["both_ends"],totals["one_end"]
        qualified=bool(full[1]/full[0]>=.8 and (full[1]-blind[1])/full[0]>=.2 and full[5]>full[6])
        if seed==101: decision=qualified
        test=evaluate(models,False)
        np.savez_compressed(R/f"val-{seed}.npz",**val)
        np.savez_compressed(R/f"test-{seed}.npz",**test)
        print(seed,"validation",[(k,v[:2].tolist()) for k,v in totals.items()],"qualified",qualified,
              "test",[(k,v.sum(0)[:2].tolist()) for k,v in test.items()],flush=True)
        if not decision: break
    save("training.json",dict(pilot_qualified=decision,columns=COLUMNS,training=metadata))
if __name__=="__main__":main()

