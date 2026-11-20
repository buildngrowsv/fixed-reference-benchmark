"""Trace donor-dependent solved queries back to their generated program and event."""
import numpy as np
from head_surgery import HOST, install_head, load_model
from run import OUT, RUNS, RoutedModel, batch, mx, restore, optimizer, write


def main():
    models = {}
    for arm in ("native","graft","removed"):
        file = "native" if arm=="native" else "graft"
        model = RoutedModel()
        assert restore(model,optimizer(),RUNS/"displaced-head-surgery"/(file+".safetensors"))==1024
        if arm=="removed":
            install_head(model,load_model(HOST))
        models[arm]=model
    rows=[]
    for i in range(32):
        tokens,y,op,changed,_=batch([482731,i])
        predictions=[np.array(m(mx.array(tokens),wrong=True).argmax(-1)) for m in models.values()]
        p,e=np.where(changed & (op>=2))
        rows.append(np.column_stack([32*i+p,e,op[p,e],y[p,e],*[v[p,e] for v in predictions]]))
    rows=np.concatenate(rows).astype(np.int32)
    np.save(OUT/"queries.npy",rows)
    native,graft,off=(rows[:,j]==rows[:,3] for j in (4,5,6))
    counts={}
    for k,label in ((2,"original"),(3,"current")):
        mask=rows[:,2]==k
        new=mask & graft & ~native
        counts[label]=dict(total=int(mask.sum()),graft_new=int(new.sum()),
                           new_lost_when_removed=int((new & ~off).sum()),
                           gained_when_removed=int((mask & ~graft & off).sum()),
                           lost_relative_to_native=int((mask & native & ~graft).sum()))
    write("result.json",dict(decision="complete",columns=["program","event","query_type","gold","native","graft","removed"],counts=counts))


if __name__=="__main__":
    main()
