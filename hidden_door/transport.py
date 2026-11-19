"""Paired cubes for learning which boundary reads to permit."""
from functools import lru_cache
import numpy as np

@lru_cache(maxsize=None)
def geometry(side,axis):
    coords=np.indices((side,)*3).reshape(3,-1).T
    offsets=np.array([[0,0,0],[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]])
    all_neighbors=(coords[:,None,:]+offsets)%side
    neighbors=np.ravel_multi_index(tuple(all_neighbors.transpose(2,0,1)),(side,)*3)
    lower=np.flatnonzero(coords[:,axis]==side//2-1)
    wrap=np.flatnonzero(coords[:,axis]==side-1)
    planes=np.concatenate((lower,wrap))
    return coords,neighbors.astype(np.int32),planes,neighbors[planes,2*axis+2]


def boards(rng,batch,side,steps,axes=(0,)):
    volume=side**3
    words=rng.choice(np.array([-1,1],np.float32),(batch,volume,32))
    commands=rng.integers(7,size=(steps,batch,volume),dtype=np.int32)
    offsets=volume*np.arange(batch,dtype=np.int32)
    index=np.empty_like(commands)
    signals=rng.integers(2,size=(batch,volume,3),dtype=np.int8)
    edited=[]
    for j in range(batch):
        axis=int(rng.choice(axes))
        coords,neighbors,lows,highs=geometry(side,axis)
        signals[j,:,0]=(coords[:,axis]>=side//2)^bool(rng.integers(2))
        # Closed latches can have either endpoint active or both inactive.
        pair=rng.integers(3,size=len(lows))
        signals[j,lows,1]=pair==1
        signals[j,highs,1]=pair==2
        k=rng.integers(side**2)
        lo,hi=lows[k],highs[k]
        signals[j,[lo,hi],1]=1
        commands[-1,j,lo]=2*axis+2
        commands[-1,j,hi]=2*axis+1
        edited.append(int(rng.choice([lo,hi])))
        index[:,j]=neighbors[np.arange(volume),commands[:,j]]+offsets[j]
    return words.reshape(-1,32),index.reshape(steps,-1),signals.reshape(-1,3),offsets+edited


def readings(index,signals):
    current=np.broadcast_to(signals,(len(index),*signals.shape))
    return np.concatenate([current,signals[index]],axis=-1)


def trace(index,signals,unrestricted=False):
    path=np.arange(index.shape[1],dtype=np.int32)
    for dest in index:
        if unrestricted: path=path[dest]
        else:
            crossing=signals[:,0]!=signals[dest,0]
            both=(signals[:,1]&signals[dest,1]).astype(bool)
            path=np.where(~crossing|both,path[dest],path)
    return path

