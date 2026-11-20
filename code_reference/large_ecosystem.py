"""Adapt equally sized early- and late-copy checkpoints to new Python projects."""
import json

from ecosystem import (Bypass, Corpus, OUT, Projects, RUNS, difference, evaluate,
                       evaluate_projects, mx, train, write)
from verify import grouped_difference

import numpy as np

BASELINE = "large-sealed-width-rate-held-out"
STEPS = 2048
RATE = .0004/3


def main():
    parent = RUNS/BASELINE
    run = json.loads((parent/"result.json").read_text())
    assert run["decision"]=="complete" and run["economical_copy"] and run["updates"]==16384
    python = Corpus()
    projects = Projects(python)
    original_files = json.loads((RUNS/"scientific-python-transfer/files.json").read_text())
    assert original_files["projects"]==projects.splits and original_files["files"]==projects.catalog
    write("files.json",original_files)
    external, retention, effects = {}, {}, {}
    for arm in ("sealed","working"):
        model = Bypass(width=1536,lane_width=192,working_exit=arm=="working")
        snapshot = mx.load(str(parent/(arm+".safetensors")))
        assert int(snapshot["step"].item())==16384
        model.load_weights([(key[8:],value) for key,value in snapshot.items() if key.startswith("weights.")])
        del snapshot
        external[arm+"_before"] = evaluate_projects(model,projects)
        train(model,arm,projects,steps=STEPS,peak_rate=RATE,checkpoint_every=256)
        external[arm+"_after"] = evaluate_projects(model,projects)
        scores,groups,windows = evaluate(model,True,python,"test",2048)
        np.savez_compressed(OUT/(arm+"-retention.npz"),scores=scores,groups=groups,windows=windows)
        with np.load(parent/(arm+"-test.npz")) as ref:
            np.testing.assert_array_equal(windows,ref["windows"])
            np.testing.assert_array_equal(groups,ref["groups"])
            before = ref["scores"][:,0]
        retention[arm] = scores[:,0]-before
        effects[arm] = dict(cpython_before=float(before.mean()/512),
                            cpython_after=float(scores[:,0].mean()/512),
                            cpython_change=grouped_difference(scores[:,0],before,groups),
                            external_before=float(external[arm+"_before"][:,3].mean()),
                            external_after=float(external[arm+"_after"][:,3].mean()))
        del model
    np.savez_compressed(OUT/"projects.npz",**external)
    original = grouped_difference(retention["sealed"],retention["working"],groups)
    other = {when:difference(external["sealed_"+when],external["working_"+when]) for when in ("before","after")}
    qualifies = (original["delta"] <= -.03 and original["interval"][1] < 0 and other["after"]["interval"][1] <= .02)
    write("result.json",dict(decision="advance" if qualifies else "complete",parent=BASELINE,steps=STEPS,peak_rate=RATE,
                            arms=effects,retention_interaction=original,external_contrasts=other))


if __name__=="__main__":
    main()
