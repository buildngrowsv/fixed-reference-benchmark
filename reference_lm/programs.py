"""Records and four-token events. The result of a query is never an input token."""
import numpy as np

def batch(seed,n=32,counterfactual=False):
    rng=np.random.default_rng(seed)
    events,registers=64,16
    initial=rng.integers(32,size=(n,registers))
    order=np.argsort(rng.random((n,registers)),axis=1)
    key=rng.integers(registers,size=(n,events))
    key[:,:registers]=order
    op=rng.choice([1,2,3],size=(n,events),p=[.55,.225,.225])
    op[:,:registers]=0
    values=rng.integers(32,size=(n,events))
    rows=np.arange(n)
    if counterfactual:
        chosen=key[:,-1].copy()
        op[:,40]=1;key[:,40]=chosen
        op[:,-1]=2+rows%2
    def encode(first):
        current=first.copy()
        val=values.copy()
        val[:,:registers]=np.take_along_axis(first,order,axis=1)
        labels=np.empty_like(key)
        changed=np.zeros_like(key,dtype=bool)
        heuristic=np.empty_like(key)
        for j in range(events):
            k=key[:,j];w=op[:,j]==1
            current[rows[w],k[w]]=val[w,j]
            labels[:,j]=np.where(op[:,j]==2,first[rows,k],current[rows,k])
            heuristic[:,j]=first[rows,k];changed[:,j]=first[rows,k]!=current[rows,k]
        tokens=np.stack([op,4+key,np.where(op<2,20+val,52),np.full_like(key,53)],axis=-1).reshape(n,-1)
        return tokens.astype(np.int32),labels.astype(np.int32),op.astype(np.int32),changed,heuristic
    one=encode(initial)
    if not counterfactual:return one
    alt=initial.copy()
    alt[rows,chosen]=(alt[rows,chosen]+rng.integers(1,32,size=n))%32
    two=encode(alt)
    assert np.all((one[0]!=two[0]).sum(1)==1)
    assert np.array_equal(one[1][:,-1]!=two[1][:,-1],op[:,-1]==2)
    return one,two

