"""Paired transfer of CPython predictors to independent numerical Python projects."""
from functools import partial
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import tarfile
import time
import urllib.request
import numpy as np
from bypass import Bypass, CodeLM, Corpus, RUNS, OUT, mx, nn, optim, evaluate, restore, snapshot, write

MANIFEST=Path(__file__).with_name("ecosystem.json")
STEPS=8192
EXCLUDED={"test","tests","testing","_vendor","vendor","vendored","extern","_lib_vendored","benchmarks","examples","_vendorized","_vendored"}

def archive(row):
    folder=RUNS.parent/"datasets/scientific-python"
    folder.mkdir(parents=True,exist_ok=True)
    target=folder/(row["sha256"]+".tar.gz")
    if not target.exists():
        tmp=target.with_suffix(f".{os.getpid()}.part")
        with urllib.request.urlopen(row["url"],timeout=45) as src,tmp.open("wb") as dest:
            while data:=src.read(1024*1024):
                dest.write(data)
        assert tmp.stat().st_size==row["bytes"] and sha256(tmp.read_bytes()).hexdigest()==row["sha256"]
        tmp.replace(target)
    assert sha256(target.read_bytes()).hexdigest()==row["sha256"]
    return target

class Projects:
    def __init__(self,python):
        self.files={};self.keys={};self.weights={};self.splits={"adapt":[],"test":[]}
        seen={sha256(raw).digest() for files in python.files.values() for _,_,raw in files}
        self.catalog=[]
        for row in json.loads(MANIFEST.read_text()):
            key=row["project"];values=[];paths=[]
            with tarfile.open(archive(row)) as tar:
                for info in sorted(tar.getmembers(),key=lambda x:x.name):
                    path=info.name.partition("/")[2]
                    parts=Path(path).parts
                    if (not info.isfile() or not path.startswith(row["package"]+"/") or
                        not path.endswith(".py") or info.size<513 or EXCLUDED.intersection(parts)
                        or parts[-1].startswith("test_") or parts[-1].endswith("_test.py")):
                        continue
                    raw=tar.extractfile(info).read();digest=sha256(raw).digest()
                    if digest in seen:continue
                    seen.add(digest);paths.append(path);values.append(np.frombuffer(raw,np.uint8))
                    self.catalog.append(dict(project=key,path=path,bytes=len(raw),sha256=digest.hex()))
            assert values, key
            self.files[key]=values;self.keys[key]=paths;self.splits[row["split"]].append(key)
            self.weights[key]=np.cumsum([len(raw)-512 for raw in values])

    def sample(self,project,rng,n):
        w=self.weights[project];points=rng.integers(w[-1],size=n)
        ix=np.searchsorted(w,points,side="right")
        off=points-np.where(ix==0,0,w[np.maximum(0,ix-1)])
        data=np.stack([self.files[project][i][o:o+513] for i,o in zip(ix,off)]).astype(np.int32)
        return data,ix,off

def evaluate_projects(model,projects):
    records=[]
    for j,project in enumerate(projects.splits["test"]):
        for i in range(32):
            data,files,offsets=projects.sample(project,np.random.default_rng([541982,j,i]),16)
            y=mx.array(data[:,1:]);z=model(mx.array(data[:,:-1]),True)
            bits=np.array(nn.losses.cross_entropy(z,y,reduction="none")).sum(1)/(512*np.log(2))
            acc=np.array((z.argmax(-1)==y).mean(1))
            records.extend(zip(np.full(16,j),files,offsets,bits,acc))
    return np.array(records)

def difference(a,b):
    np.testing.assert_array_equal(a[:,:3],b[:,:3])
    project=a[:,0].astype(int);diff=a[:,3]-b[:,3];rng=np.random.default_rng(282471)
    means=[];draws=[]
    for i in np.unique(project):
        use=project==i
        files, inverse=np.unique(a[use,1],return_inverse=True)
        numer=np.bincount(inverse,weights=diff[use]);denom=np.bincount(inverse)
        sample=rng.integers(len(files),size=(3000,len(files)))
        draws.append(numer[sample].sum(1)/denom[sample].sum(1));means.append(float(diff[use].mean()))
    return dict(project_differences=means,macro=float(np.mean(means)),interval=np.quantile(np.mean(draws,axis=0),[.025,.975]).tolist())

def train(model,arm,projects,rehearsal=None,*,steps=STEPS,peak_rate=.0002,checkpoint_every=64):
    opt=optim.AdamW(lambda t:peak_rate*mx.minimum(1.,(t.astype(mx.float32)+1)/128),weight_decay=.03,bias_correction=True)
    path=OUT/(arm+".safetensors")
    def objective(m,x,y):return nn.losses.cross_entropy(m(x,True),y,reduction="mean")
    start=restore(model,opt,path) if path.exists() else 0
    vg=nn.value_and_grad(model,objective);state=[model.state,opt.state]
    @partial(mx.compile,inputs=state,outputs=state)
    def step(x,y):
        loss,grad=vg(model,x,y);grad,_=optim.clip_grad_norm(grad,1.);opt.update(model,grad);return loss
    since=time.monotonic()
    for t in range(start+1,steps+1):
        rng=np.random.default_rng([638151,t]);names=projects.splits["adapt"]
        if rehearsal is not None and t%8==0:
            data,*_=rehearsal.batch("train",[629147,t],16)
        else:
            data,_,_=projects.sample(names[rng.integers(len(names))],rng,16)
        loss=step(mx.array(data[:,:-1]),mx.array(data[:,1:]));mx.eval(loss,state)
        if t%64==0 or t==steps:
            if not np.isfinite(loss.item()):raise ValueError("nonfinite training loss")
            if t%checkpoint_every==0 or t==steps:
                snapshot(model,opt,t,path)
            write("progress.json",dict(arm=arm,step=t,target=steps,time=time.time(),loss=loss.item(),tokens_per_second=(t-start)*8192/(time.monotonic()-since)))

def main():
    smoke=sys.argv[1:]==["smoke"]
    if smoke:mx.set_default_device(mx.cpu)
    python=Corpus();projects=Projects(python)
    if smoke:
        for name,v in projects.files.items():
            data,ix,off=projects.sample(name,np.random.default_rng(881),3)
            assert data.shape==(3,513) and len(ix)==len(off)==3
            print(name,len(v),sum(map(len,v)))
        return
    write("files.json",dict(projects=projects.splits,files=projects.catalog))
    scores={};retention={}
    for arm,klass,source in (("sealed",Bypass,"sealed-bypass-main/bypass.safetensors"),
                              ("readable",CodeLM,"cpython-reference-main/fixed.safetensors")):
        model=klass();parent=RUNS/source
        data=mx.load(str(parent));assert int(data["step"].item())==16384
        model.load_weights([(k[8:],v) for k,v in data.items() if k.startswith("weights.")]);del data
        scores[arm+"_before"]=evaluate_projects(model,projects)
        train(model,arm,projects)
        scores[arm+"_after"]=evaluate_projects(model,projects)
        v,g,w=evaluate(model,True,python,"test",2048)
        np.savez_compressed(OUT/(arm+"-retention.npz"),scores=v,groups=g,windows=w)
        ref=RUNS/("sealed-bypass-main/test.npz" if arm=="sealed" else "cpython-reference-main/fixed-test.npz")
        with np.load(ref) as old:
            for k,current in (("groups",g),("windows",w)):np.testing.assert_array_equal(old[k],current)
            retention[arm]=float((v[:,0]-old["scores"][:,0]).mean()/512)
    np.savez_compressed(OUT/"projects.npz",**scores)
    contrasts={t:difference(scores["sealed_"+t],scores["readable_"+t]) for t in ("before","after")}
    means={k:float(v[:,3].mean()) for k,v in scores.items()}
    metric=contrasts["after"]
    success=metric["interval"][1]<.02 and sum(d<.02 for d in metric["project_differences"])>=5
    write("result.json",dict(decision="complete",within_margin=bool(success),projects=projects.splits["test"],bpb=means,contrasts=contrasts,cpython_loss_change=retention))

if __name__=="__main__":main()
