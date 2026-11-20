from collections import Counter
from hashlib import sha256
from pathlib import Path
import re
import tarfile
import numpy as np

ARCHIVE=Path.home()/"research/routed-expert-splice/cache/cpython.tar.gz"
DIGEST="37c8e18b35e659da6ff4db74caea50ed4170cf66c6d51f97d22e436e338d7b5b"
LENGTH=512
def group(path):
    bits=path.split("/")[1:]
    stem=bits[0].removesuffix(".py")
    if stem=="test" and len(bits)>1:
        stem=bits[1].removesuffix(".py").removeprefix("test_")
    return stem

class Corpus:
    def __init__(self):
        assert sha256(ARCHIVE.read_bytes()).hexdigest()==DIGEST
        self.files={s:[] for s in ("train","val","test")}
        hashes=set()
        with tarfile.open(ARCHIVE) as archive:
            for info in sorted(archive.getmembers(),key=lambda x:x.name):
                path=info.name.partition("/")[2]
                if not (info.isfile() and path.startswith("Lib/") and path.endswith(".py") and info.size>=513):continue
                g=group(path)
                bucket=int.from_bytes(sha256(("code-ref-v1/"+g).encode()).digest()[:4],"big")%100
                split="train" if bucket<80 else "val" if bucket<90 else "test"
                source=archive.extractfile(info).read()
                digest=sha256(source).digest()
                if digest in hashes:continue
                hashes.add(digest)
                self.files[split].append((path,g,np.frombuffer(source,dtype=np.uint8)))
        self.weights={};self.unigram=np.ones(256)
        for split,fl in self.files.items():
            w=np.array([len(x)-LENGTH for _,_,x in fl]);self.weights[split]=np.cumsum(w)
            if split=="train":
                for _,_,x in fl:self.unigram+=np.bincount(x,minlength=256)
        self.unigram/=self.unigram.sum()
    def batch(self,split,seed,n):
        rng=np.random.default_rng(seed)
        weights=self.weights[split];points=rng.integers(weights[-1],size=n)
        idx=np.searchsorted(weights,points,side="right")
        offsets=points-np.where(idx==0,0,weights[np.maximum(0,idx-1)])
        rows=[self.files[split][i] for i in idx]
        b=np.stack([x[o:o+LENGTH+1] for (_,_,x),o in zip(rows,offsets)]).astype(np.int32)
        return b,np.array([g for _,g,_ in rows]),idx,offsets
    def manifest(self):
        return {s:[dict(path=p,group=g,bytes=len(x)) for p,g,x in rows] for s,rows in self.files.items()}

def repeats(row):
    seen={};mask=np.zeros(LENGTH,dtype=bool)
    for m in re.finditer(rb"(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z_0-9]{3,}(?![A-Za-z0-9_])",row.astype(np.uint8).tobytes()):
        word=m.group();pos=m.start()
        if pos-seen.get(word,pos)>=64 and m.end()<=len(row)-1:mask[m.end()-2]=True
        seen.setdefault(word,pos)
    return mask

def suffix_scores(row,unigram):
    # At position i vote only from suffix occurrences whose continuation has already been observed.
    b=row.astype(np.uint8).tobytes();tables=[{} for _ in range(9)]
    losses=np.empty(LENGTH);hits=np.empty(LENGTH,dtype=bool)
    for i in range(LENGTH):
        if i:
            for k in range(1,min(8,i)+1):
                key=b[i-k:i]
                tables[k].setdefault(key,Counter())[b[i]]+=1
        nxt=b[i+1];counts={}
        for k in range(min(i+1,8),0,-1):
            counts=tables[k].get(b[i-k+1:i+1],{})
            if counts:break
        p=(counts.get(nxt,0)+unigram[nxt])/(1+sum(counts.values()))
        top=max(counts,key=lambda x:(counts[x]+unigram[x],-x)) if counts else int(unigram.argmax())
        losses[i]=-np.log2(p);hits[i]=top==nxt
    return losses,hits

