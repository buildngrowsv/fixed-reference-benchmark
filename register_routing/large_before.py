"""Score both original 512-update checkpoints on the fresh continuation panel."""
import numpy as np
from run import Model, RoutedModel, RUNS, OUT, optimizer, restore, evaluate, scores, write
from large_revisit import WIDTH


def main():
    rows = {}
    for arm, architecture in (("dense",Model),("rotated",RoutedModel)):
        model = architecture(WIDTH)
        assert restore(model,optimizer(),RUNS/"large-register-dense-vs-displaced"/(arm+".safetensors"))==512
        rows[arm] = evaluate(model,arm,427361)
        del model
        later = np.load(RUNS/"large-register-quarter-rate"/(arm+"-counts.npy"))
        np.testing.assert_array_equal(rows[arm][:,[0,2,3,5]],later[:,[0,2,3,5]])
    np.savez_compressed(OUT/"counts.npz",**rows)
    write("result.json",dict(decision="complete",panel_seed=427361,updates=512,accuracy={k:scores(v) for k,v in rows.items()}))


if __name__=="__main__":
    main()
