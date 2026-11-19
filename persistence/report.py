"""Whole-word counts, with a cluster interval conditional on the single pilot fit."""
import json
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent / "results"
records, paired = {}, {}
for name in ("assigned", "biased", "neutral", "persistent"):
    meta = json.loads((R/f"{name}-101.json").read_text())
    bank = meta["fresh_address"]
    with np.load(R/f"{name}-101.npz") as data:
        metrics = {k: [int(i) for i in data[v].sum(0)[:2]] for k,v in
                   (("primary","primary"), ("undisturbed","from_memory"),
                    ("selected_exchanged",f"corrupt{bank}"), ("other_exchanged",f"corrupt{1-bank}"))}
        paired[name] = data["primary"][:,:2].reshape(64,2,2).sum(1)
    records[name] = dict(parameters=meta["params"],bank=bank,metrics_denominator_then_hits=metrics,
                         validation=meta["fresh_validation"]["primary"],training_seconds=meta["train_seconds"],
                         mlx_peak_bytes=meta["peak_bytes"])
d = paired["persistent"][:,1] - paired["neutral"][:,1]
n = paired["neutral"][:,0]
rng = np.random.default_rng(7165)
draw = rng.integers(64,size=(5000,64))
result = dict(arms=records,persistent_minus_neutral=dict(difference=float(d.sum()/n.sum()),
              paired_volume_interval=np.quantile(d[draw].sum(1)/n[draw].sum(1),[.025,.975]).tolist(),
              scope="64 volume pairs conditional on training seed 101"))
(R/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))

