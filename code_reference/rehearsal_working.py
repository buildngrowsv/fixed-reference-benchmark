"""Repeat the replay intervention in the same-size working-only model."""
import numpy as np

from ecosystem import Bypass, Corpus, Projects, RUNS, OUT, mx, train, evaluate, evaluate_projects, difference, write
from verify import grouped_difference


def main():
    python = Corpus()
    projects = Projects(python)
    working = Bypass(working_exit=True)
    checkpoint = mx.load(str(RUNS/"working-exit-main/bypass.safetensors"))
    assert int(checkpoint["step"].item()) == 16384
    working.load_weights([(k[8:], v) for k,v in checkpoint.items() if k.startswith("weights.")])
    del checkpoint
    train(working, "working-rehearsal", projects, rehearsal=python)
    external = evaluate_projects(working, projects)
    py_scores, groups, windows = evaluate(working, True, python, "test", 2048)
    np.savez_compressed(OUT/"cpython.npz", scores=py_scores, groups=groups, windows=windows)
    np.savez_compressed(OUT/"external.npz", scores=external)
    no_replay = RUNS/"scientific-python-equal-size"
    sealed_replay = RUNS/"sealed-python-rehearsal"
    earlier = np.load(no_replay/"projects.npz")["after"]
    sealed_old = np.load(RUNS/"scientific-python-transfer/projects.npz")["sealed_after"]
    sealed_new = np.load(sealed_replay/"external.npz")["scores"]
    for v in (earlier,sealed_old,sealed_new):
        np.testing.assert_array_equal(v[:,:3], external[:,:3])
    interaction_baseline = earlier.copy()
    interaction_baseline[:,3] += sealed_new[:,3] - sealed_old[:,3]
    ordinary = np.load(no_replay/"working-retention.npz")
    sealed_a = np.load(RUNS/"scientific-python-transfer/sealed-retention.npz")
    sealed_b = np.load(sealed_replay/"cpython.npz")
    for v in (ordinary,sealed_a,sealed_b):
        np.testing.assert_array_equal(v["groups"],groups)
        np.testing.assert_array_equal(v["windows"],windows)
    ext = difference(external,earlier)
    cpy = grouped_difference(py_scores[:,0],ordinary["scores"][:,0],groups)
    write("result.json",dict(decision="complete",works_without_early_copy=bool(ext["interval"][1]<0 and cpy["interval"][1]<0),
          external=ext, cpython=cpy, external_interaction=difference(external,interaction_baseline),
          cpython_interaction=grouped_difference(py_scores[:,0]-ordinary["scores"][:,0],
                                                sealed_b["scores"][:,0]-sealed_a["scores"][:,0],groups),
          external_bpb=float(external[:,3].mean()),cpython_bpb=float(py_scores[:,0].mean()/512)))


if __name__=="__main__":
    main()
