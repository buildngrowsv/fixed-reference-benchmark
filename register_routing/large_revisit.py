"""Retest the dense control and displaced memory read in a wider host."""
from functools import partial
import time
import numpy as np
from mlx.utils import tree_flatten
from run import Model,RoutedModel,OUT,RUNS,batch,evaluate,mx,nn,optim,optimizer,restore,scores,snapshot,uncertainty,write
from relay import exchange

WIDTH=1536

class NonfiniteTrainingLoss(ValueError):
    def __init__(self, arm, step):
        self.arm, self.step = arm, step
        super().__init__(f"{arm} had nonfinite training loss at update {step}")


def fit(arm,target=512,parent=None,learning_rate=None):
    mx.random.seed(101)
    model=Model(WIDTH) if arm=="dense" else RoutedModel(WIDTH)
    opt=optimizer(learning_rate)
    path=OUT/(arm+".safetensors")
    start=0
    if path.exists():
        start=restore(model,opt,path)
    elif parent is not None:
        start=restore(model,opt,parent/path.name)
        assert start==512
    if learning_rate is not None:
        # Restoring Adam moments also restores the saved scalar step size.
        opt.learning_rate = learning_rate
    def objective(m,x,y,ops):
        logits=m(x,False) if arm=="dense" else m(x,wrong=True)
        use=ops>=2
        loss=nn.losses.cross_entropy(logits,y,reduction="none")
        return (loss*use).sum()/mx.maximum(use.sum(),1)
    vg=nn.value_and_grad(model,objective)
    state=[model.state,opt.state]
    @partial(mx.compile,inputs=state,outputs=state)
    def step(x,y,ops):
        loss,g=vg(model,x,y,ops)
        g,_=optim.clip_grad_norm(g,1.)
        opt.update(model,g)
        return loss
    count=sum(v.size for _,v in tree_flatten(model.parameters()))
    clock=time.monotonic()
    for t in range(start+1,target+1):
        ids,y,ops,_,_=batch([101,27491,t])
        loss=step(mx.array(ids),mx.array(y),mx.array(ops))
        mx.eval(loss,state)
        if t==start+1 and learning_rate is not None:
            np.testing.assert_allclose(opt.learning_rate.item(),learning_rate,rtol=1e-6)
        if t==1 or t%16==0:
            if not np.isfinite(loss.item()):raise NonfiniteTrainingLoss(arm,t)
            if t%256==0:snapshot(model,opt,t,path)
            write("progress.json",dict(arm=arm,step=t,target=target,parameters=count,loss=loss.item(),time=time.time(),
                  tokens_per_second=(t-start)*8192/(time.monotonic()-clock)))
    return model,count

def main():
    with np.load(RUNS/"register-addressing/test.npz") as saved:
        counts={"small_"+k:saved[k] for k in ("dense","rotated")}
    meta={}
    for arm in ("dense","rotated"):
        model,meta[arm]=fit(arm)
        counts["large_"+arm]=evaluate(model,arm,625137)
        if arm=="rotated":swap=exchange(model,"large-exchange",1,152843,64)
        del model
    for a in counts.values():np.testing.assert_array_equal(a[:,[0,2,3,5]],counts["small_dense"][:,[0,2,3,5]])
    np.savez_compressed(OUT/"counts.npz",**counts)
    acc={k:scores(v) for k,v in counts.items()}
    gaps={s:float(np.mean(acc[s+"_rotated"])-np.mean(acc[s+"_dense"])) for s in ("small","large")}
    interaction=counts["large_rotated"].copy()
    for col in (1,4):interaction[:,col]-=counts["small_rotated"][:,col]-counts["small_dense"][:,col]
    write("result.json",dict(decision="complete",parameters=meta,width=WIDTH,accuracy=acc,gaps=gaps,
          large_gap_interval=uncertainty(counts["large_rotated"],counts["large_dense"]),
          change_in_gap=gaps["large"]-gaps["small"],change_interval=uncertainty(interaction,counts["large_dense"]),swap=swap))

if __name__=="__main__":main()
