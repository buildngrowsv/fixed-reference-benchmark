"""Reuse the frozen one-hop learned locator across three reads and two host architectures."""
from functools import partial
from pathlib import Path
from typing import NamedTuple
import hashlib
import json
import os
import time

from learned_address import (ALPHABET, PAYLOADS, Locator, Reader, atomic_json, ci, fit, make_batch,
                             mx, nn, np, optim, OUT, predict)

BATCH = 64
SOURCE_SEED = 79222
TARGET_STEPS = 128
ARCHIVE = Path(__file__).resolve().parent / "address_study/results/paired-101.npz"


class Records(NamedTuple):
    a_keys: np.ndarray
    a_links: np.ndarray
    b_keys: np.ndarray
    b_links: np.ndarray
    c_keys: np.ndarray
    payloads: np.ndarray
    query: np.ndarray
    kind: np.ndarray


def program(seed, count, n, offset):
    rng = np.random.default_rng(seed)
    pool = ALPHABET[ALPHABET.sum(axis=1) % 2 == 0]
    names = [pool[np.argsort(rng.random((count, len(pool))), axis=1)[:, :n]] for _ in range(3)]
    row = np.arange(count)[:, None]

    def links(keys, dest):
        mapping = np.argsort(rng.random((count, n)), axis=1)
        previous = np.roll(mapping, 1, axis=1)
        # Sorted times retain each name's old/current order, with everything interleaved.
        times = np.sort(rng.random((count, 2, n)), axis=1)
        order = np.argsort(times.reshape(count, 2*n), axis=1)
        assigned = np.concatenate((previous, mapping), axis=1)
        return keys[row, order%n], dest[row, assigned[row, order]]

    a, al = links(*names[:2])
    b, bl = links(*names[1:])
    order = np.tile(np.arange(n),3)[np.argsort(rng.random((count,3*n)), axis=1)]
    c = names[2][row,order]
    v = np.argsort(rng.random((count,PAYLOADS)),axis=1)[:,:3*n] + offset
    query = names[0][row,np.argsort(rng.random((count,n)),axis=1)[:,:8]]
    return Records(a,al,b,bl,c,v,np.repeat(query,2,axis=1),np.tile([0,1],(count,8)))


def oracle(p):
    def read(table, key, late):
        matching = (table[:,None,:,:]==key[:,:,None,:]).all(axis=-1)
        assert matching.any(axis=2).all()
        first = matching.argmax(2)
        last = table.shape[1]-1-matching[:,:,::-1].argmax(2)
        return np.where(late,last,first)
    row=np.arange(len(p.query))[:,None]
    link=read(p.a_keys,p.query,True)
    mid=read(p.b_keys,p.a_links[row,link],True)
    location=read(p.c_keys,p.b_links[row,mid],p.kind==1)
    return link,location,p.payloads[row,location]


def counterfactuals(p):
    p=p._replace(query=p.query[:,:2],kind=p.kind[:,:2])
    row=np.arange(len(p.query))
    current,_,expected=oracle(p)
    current=current[:,0]
    obsolete=(p.a_keys==p.query[:,0:1]).all(2).argmax(1)
    assert np.all(obsolete<current)
    next_key=p.a_links[row,(current+1)%p.a_links.shape[1]].copy()
    used=p.a_links[row,current]
    old=p.a_links[row,obsolete]
    # Change either record by one real edit. Current maps are bijective.
    invalid=(next_key==used).all(1)|(next_key==old).all(1)
    if invalid.any():
        choices=p.a_links[invalid]
        eligible=(choices!=used[invalid,None,:]).any(2)&(choices!=old[invalid,None,:]).any(2)
        assert eligible.any(1).all()
        next_key[invalid]=choices[np.arange(len(choices)),eligible.argmax(1)]
    variants=[]
    for location in (current,obsolete):
        changed=p.a_links.copy()
        assert (changed[row,location]!=next_key).any(1).all()
        changed[row,location]=next_key
        variants.append(p._replace(a_links=changed))
    assert np.all(oracle(variants[0])[2]!=expected)
    np.testing.assert_array_equal(oracle(variants[1])[2],expected)
    return variants


class Convolution(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv1d(32,32,3,padding=1)
        self.second=nn.Conv1d(32,32,3,padding=1)

    def __call__(self, x):
        return x + self.second(nn.gelu(self.conv(x)))


class Recurrent(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru=nn.GRU(32,32)

    def __call__(self, x):
        return x + self.gru(x)


class Host(nn.Module):
    def __init__(self, encoder, offset):
        super().__init__()
        self.locator=Locator()
        self.offset=offset
        self.embedding=nn.Embedding(PAYLOADS,32)
        self.encoder=encoder()
        self.output=nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Linear(64,PAYLOADS))

    def weights(self, p):
        a,al,b,bl,c,_,q,kind=p
        shared=self.locator
        first=mx.softmax(shared(a,q,mx.ones_like(kind)),axis=-1)
        second=mx.softmax(shared(b,al,mx.ones(al.shape[:2],dtype=mx.int32)),axis=-1)
        prefix=first@second
        first_write=mx.softmax(shared(c,bl,mx.zeros(bl.shape[:2],dtype=mx.int32)),axis=-1)
        last_write=mx.softmax(shared(c,bl,mx.ones(bl.shape[:2],dtype=mx.int32)),axis=-1)
        return mx.where(kind[...,None]==0,prefix@first_write,prefix@last_write)

    def decode(self, w, v):
        data=self.encoder(self.embedding(v-self.offset))
        return self.output(w@data)

    def __call__(self, p):
        return self.decode(self.weights(p),p[5])


def to_mx(p):
    return tuple(mx.array(x,dtype=mx.int32) for x in p)


def train(model, seed, arm, steps=TARGET_STEPS):
    opt=optim.Adam(.003,bias_correction=True)

    def objective(m,p,answer):
        return nn.losses.cross_entropy(m(p),answer,reduction="mean")

    gradient=nn.value_and_grad(model,objective)
    state=[model.state,opt.state]

    @partial(mx.compile,inputs=state,outputs=state)
    def step(p,answer):
        loss,grad=gradient(model,p,answer)
        opt.update(model,grad)
        return loss

    clock=time.monotonic()
    for i in range(1,steps+1):
        p=program([seed,i],BATCH,12,model.offset)
        e=step(to_mx(p),mx.array(oracle(p)[2]-model.offset,dtype=mx.int32))
        mx.eval(e,state)
        if i==1 or i%32==0:
            if not np.isfinite(e.item()):
                raise ValueError(f"nonfinite training loss: {arm}, step {i}")
            atomic_json("progress.json",dict(arm=arm,step=i,target=steps,loss=e.item(),time=time.time()))
    return time.monotonic()-clock


def source(seed, arm):
    mx.random.seed(seed)
    reader=Reader(0)
    fit(reader,768,SOURCE_SEED,arm)
    return reader


def verify_donor(donor):
    with np.load(ARCHIVE) as original:
        positions=[]
        answers=[]
        for i in range(16):
            b=make_batch([810731+101,i],64,True)
            address,answer,_,_=predict(donor,b)
            positions.append(address)
            answers.append(answer)
        for k,v in (("donor_p",positions),("donor_answers",answers)):
            np.testing.assert_array_equal(np.concatenate(v),original[k])
    return hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()


def predict_host(model, p, roll=False):
    inputs=to_mx(p)
    weights=model.weights(inputs)
    if roll:
        weights=mx.concatenate((weights[...,-7:],weights[...,:-7]),axis=-1)
    answer=np.array(model.decode(weights,inputs[5]).argmax(-1))+model.offset
    return np.array(weights.argmax(-1)),answer


def evaluate(models, offset, seed, n, n_batches):
    arrays={}
    for j in range(n_batches):
        p=program([seed,j],BATCH,n,offset)
        _,location,expected=oracle(p)
        row=dict(gold=expected,address=location)
        for arm,m in models.items():
            a,b=predict_host(m,p,arm=="rolled")
            row[arm+"_address"]=a
            row[arm+"_answer"]=b
        if n==20:
            changed,stale=counterfactuals(p)
            for name,variant in (("current",changed),("stale",stale)):
                row[name+"_gold"]=oracle(variant)[2]
                row[name+"_answer"]=predict_host(models["graft"],variant)[1]
            for field,data in zip(Records._fields,p):
                row["input_"+field]=data
            row["edited_link"]=changed.a_links
            row["stale_link"]=stale.a_links
        for key,data in row.items():
            # All retained keys, positions, and remapped payloads fit in one byte.
            arrays.setdefault(key,[]).append(data.astype(np.uint8))
    return {k:np.concatenate(v) for k,v in arrays.items()}


def summarize(arrays):
    labels=arrays["gold"]
    hits={}
    report={}
    for key in ("graft","scratch","pretrained","rolled"):
        correct=arrays[key+"_answer"]==labels
        hits[key]=correct
        report[key]=dict(accuracy=float(correct.mean()),
            address_accuracy=float((arrays[key+"_address"]==arrays["address"]).mean()))
    result=dict(arms=report,programs=len(labels),queries=labels.size,
        contrasts={k:dict(delta=float(diff.mean()),interval=ci(diff))
            for k in ("scratch","pretrained","rolled")
            for diff in [hits["graft"].mean(1)-hits[k].mean(1)]})
    if "current_answer" in arrays:
        clean=hits["graft"][:,:2]
        result["causal"]={}
        for name in ("current","stale"):
            ok=arrays[name+"_answer"]==arrays[name+"_gold"]
            result["causal"][name]=dict(clean_correct=int(clean.sum()),both_correct=int((ok&clean).sum()),
                correct_unconditional=int(ok.sum()),all_queries=ok.size)
    return result


def verdict(test):
    con=test["contrasts"]
    win=all(con[k]["delta"]>=.03 and con[k]["interval"][0]>0 for k in ("scratch","pretrained"))
    causal=all(v["clean_correct"] and v["both_correct"]/v["clean_correct"]>=.90 for v in test["causal"].values())
    return bool(win and causal and con["rolled"]["interval"][0]>0)


def count(model):
    from mlx.utils import tree_flatten
    return sum(x.size for _,x in tree_flatten(model.parameters()))


def main():
    mx.set_default_device(mx.cpu)
    mx.set_cache_limit(64*1024*1024)
    start=time.monotonic()
    donated=source(101,"reconstruct one-hop donor")
    witness=verify_donor(donated)
    donor=donated.locator
    donor.freeze()
    from mlx.utils import tree_flatten
    frozen=[np.array(v).copy() for _,v in tree_flatten(donor.parameters())]
    report={}
    for seed,pretrain_seed in ((503,341),(607,861)):
        native=source(pretrain_seed,f"{seed}: independent native one-hop")
        for name,encoder,offset in (("convolution",Convolution,64),("recurrent",Recurrent,128)):
            arms={}
            elapsed={}
            for arm in ("graft","scratch","pretrained"):
                mx.random.seed(seed)
                m=Host(encoder,offset)
                if arm=="graft":
                    m.locator=donor
                if arm=="pretrained":
                    m.locator.update(native.locator.parameters())
                elapsed[arm]=train(m,42918+seed,f"{name}/{seed}/{arm}")
                arms[arm]=m
            arms["rolled"]=arms["graft"]
            test=evaluate(arms,offset,83639+seed,20,16)
            familiar=evaluate(arms,offset,836390+seed,12,4)
            np.savez_compressed(OUT/f"{name}-{seed}.npz",**test)
            result=summarize(test)
            report[f"{name}-{seed}"]=dict(held_out=result,familiar=summarize(familiar),
                passes=verdict(result),weights=count(arms["graft"]),seconds=elapsed)
    for after,before in zip(tree_flatten(donor.parameters()),frozen):
        np.testing.assert_array_equal(after[1],before)
    atomic_json("result.json",dict(decision="complete",donor_archive_sha256=witness,
        donor_weights=count(donor),source_steps=768,target_steps=TARGET_STEPS,fits=report,
        advantage_replicates=all(p["passes"] for p in report.values()),
        elapsed_seconds=time.monotonic()-start,device=str(mx.default_device()),hardware=mx.device_info()["device_name"]))


if __name__=="__main__":
    main()
