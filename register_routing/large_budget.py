"""Is the apparent loss of the routed advantage at 227M a short-budget result?"""
import numpy as np
from large_revisit import fit,WIDTH
from run import Model,RoutedModel,RUNS,OUT,optimizer,restore,evaluate,scores,uncertainty,write
from relay import exchange

def main():
    parent=RUNS/"large-register-dense-vs-displaced"
    rows={};swap=None
    for arm,klass in (("dense",Model),("rotated",RoutedModel)):
        model=klass(WIDTH)
        assert restore(model,optimizer(),parent/(arm+".safetensors"))==512
        rows[arm+"_before"]=evaluate(model,arm,427361)
        del model
        model,count=fit(arm,4096,parent)
        rows[arm+"_after"]=evaluate(model,arm,427361)
        if arm=="rotated" and scores(rows[arm+"_after"])[0]>=.50:
            swap=exchange(model,"long-exchange",1,873614,64)
        del model
    for a in rows.values():np.testing.assert_array_equal(a[:,[0,2,3,5]],rows["dense_before"][:,[0,2,3,5]])
    np.savez_compressed(OUT/"counts.npz",**rows)
    accuracy={k:scores(v) for k,v in rows.items()}
    after=rows["rotated_after"];dense=rows["dense_after"]
    gap=float(np.mean(scores(after))-np.mean(scores(dense)));ci=uncertainty(after,dense)
    write("result.json",dict(decision="complete",parameter_count=count,updates=4096,accuracy=accuracy,
          large_gap=gap,interval=ci,regained_benefit=bool(gap>=.1 and ci[0]>0),value_exchange=swap))

if __name__=="__main__":main()
