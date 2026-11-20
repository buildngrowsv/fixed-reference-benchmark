"""Verify portability and schedule shift with regenerated program labels."""
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
sys.path.insert(0, str(HERE.parents[1] / "reference_lm"))
from programs import batch


def bootstrap(arms, coefficients):
    n = len(arms[0])
    base = arms[0][:, [0, 3]]
    changes = sum(c * a[:, [1, 4]] for c, a in zip(coefficients, arms))
    rng = np.random.default_rng(192841)
    draws = []
    for _ in range(40):
        ids = rng.integers(n, size=(100, n))
        draws.extend((changes[ids].sum(1) / base[ids].sum(1)).mean(1))
    return [(changes.sum(0)/base.sum(0)).mean(), *np.quantile(draws,[.025,.975])]


def counts_check(panel, seed, orders=None):
    rows = []
    for i in range(len(next(iter(panel.values())))//32):
        x, y, op, changed, guess = batch([seed,i])
        if orders is not None:
            for tokens, labels, perm in zip(x, y, orders[32*i:32*(i+1)]):
                events = tokens.reshape(64,4)
                np.testing.assert_array_equal(sorted(perm), np.arange(64))
                assert perm[-1]==63
                original, current = {}, {}
                for j in perm:
                    typ, reg, num, _ = events[j]
                    if typ==0:
                        assert reg not in original
                        original[reg] = num-20
                    if typ<2:
                        current[reg] = num-20
                    if typ>=2:
                        assert (original[reg] if typ==2 else current[reg]) == labels[j]
                for reg in original:
                    assert np.all(np.diff(perm[events[perm,1]==reg].astype(int))>0)
        cols=[]
        for kind in (2,3):
            use = changed & (op==kind)
            cols.extend([use.sum(1),((guess==y)&use).sum(1)])
        rows.extend(np.column_stack(cols))
    for a in panel.values():
        np.testing.assert_array_equal(a[:,[0,2,3,5]],rows)
        assert np.all(a[:,[1,4]]<=a[:,[0,3]]) and np.all(a>=0)


def main():
    folder = RESULTS / "compact-portability"
    e = np.load(folder/"evidence.npz")
    report = json.loads((folder/"result.json").read_text())
    targets = []
    for i in range(128):
        a,b = batch([35719,i],32,True)
        targets.extend(np.column_stack([a[1][::2,-1],b[1][::2,-1]]))
    pairs = e["pairs"]
    np.testing.assert_array_equal(pairs[:,0],targets)
    for arm,k in (("compact",1),("full",4)):
        gold=pairs[:,0]
        good=(pairs[:,k]==gold).all(1)
        hits=[good & (pairs[:,j]==gold[:,::-1]).all(1) for j in (k+1,k+2)]
        r=report["pairs"][arm]
        assert [good.sum(),*[x.sum() for x in hits]] == [r[t] for t in ("clean","source","control")]
        effect=hits[0].astype(float)-hits[1]
        ids=np.random.default_rng(291637).integers(len(gold),size=(4000,len(gold)))
        np.testing.assert_allclose(np.quantile(effect[ids].mean(1),[.025,.975]),r["interval"])
    counts_check({k:e[k] for k in ("compact","full")},272461)
    gap=bootstrap([e["compact"],e["full"]],[1,-1])
    np.testing.assert_allclose(gap[1:],report["compact_minus_full_interval"])
    shift=RESULTS/"interleaved"
    raw=np.load(shift/"scores.npz")
    r=json.loads((shift/"result.json").read_text())
    scored={k:raw[k] for k in raw if k!="orders"}
    counts_check(scored,270643,raw["orders"])
    for name in ("hybrid","displaced","exact"):
        expected=r["paired"][name]
        np.testing.assert_allclose(bootstrap([raw[name+"_original"],raw[name+"_interleaved"]],[1,-1]),
                                   [expected["loss"],*expected["interval"]])
    excess=bootstrap([raw[k] for k in ("hybrid_original","hybrid_interleaved","exact_original","exact_interleaved")],[1,-1,-1,1])
    np.testing.assert_allclose(excess,[r["excess_hybrid_loss_over_exact"]["delta"],*r["excess_hybrid_loss_over_exact"]["interval"]])
    print(json.dumps(dict(portable_recall_gap=gap,specific_answer_swaps=report["pairs"],schedule_programs=len(raw["orders"]),
                           excess_hybrid_schedule_loss=excess),indent=2))


if __name__=="__main__":
    main()
