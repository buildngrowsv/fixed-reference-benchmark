"""Check each answer, lost solve, and paired intervention from the graft ledger."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "results" / "large-recipient"
sys.path.insert(0, str(HERE.parents[1] / "reference_lm"))
from programs import batch


def main():
    ledger = np.load(DATA / "queries.npy")
    report = json.loads((DATA / "result.json").read_text())
    rows = np.load(DATA / "counts.npz")
    expected = []
    for i in range(32):
        _, gold, op, changed, _ = batch([518263,i])
        program, event = np.where(changed & (op>=2))
        expected.extend(np.column_stack((32*i+program, event, op[program,event], gold[program,event])))
    np.testing.assert_array_equal(ledger[:,:4], expected)
    program, _, op, gold = ledger[:,:4].T
    native, aligned, permuted, off = (ledger[:,4:] == gold[:,None]).T
    for arm, solved in zip(("native","aligned","permuted","donor_off"), (native,aligned,permuted,off)):
        for j, kind in enumerate((2,3)):
            mask = op==kind
            np.testing.assert_array_equal(np.bincount(program[mask & solved],minlength=1024), rows[arm][:,3*j+1])
            np.testing.assert_allclose(solved[mask].mean(),report["accuracy"][arm][j])
    diffs = dict(graft_new=int((aligned & ~native).sum()), new_lost_off=int((aligned & ~native & ~off).sum()),
                 native_solves_lost=int((~aligned & native).sum()), off_rescues=int((~aligned & off).sum()))
    assert all(report[k]==v for k,v in diffs.items())
    delta = (rows["aligned"]-rows["donor_off"])[:,[1,4]]
    totals = rows["aligned"][:,[0,3]]
    rng = np.random.default_rng(192841)
    draws = []
    for _ in range(80):
        ix = rng.integers(1024,size=(50,1024))
        draws.extend((delta[ix].sum(1)/totals[ix].sum(1)).mean(1))
    interval = np.quantile(draws,[.025,.975])
    np.testing.assert_allclose(interval,report["effects"]["donor_off"]["interval"])
    print(json.dumps(dict(queries=len(ledger),all_on_to_off_losses=int((aligned & ~off).sum()),
                          paired_off_interval=interval.tolist(),**diffs),indent=2))


if __name__ == "__main__":
    main()
