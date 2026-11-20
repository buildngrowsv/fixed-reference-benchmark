"""Contrast decoding at the source with the model's paired answers."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

here = Path(__file__).resolve().parent
d = json.loads((here/"audit.json").read_text())
plt.style.use("dark_background")
fig,ax = plt.subplots(figsize=(8.5,3.9),layout="constrained",facecolor="#101916")
ax.set_facecolor("#101916")
labels = ["Read the fixed slice","Read the neighboring slice","Swap the fixed slice","Swap the neighboring slice","Swap all 1,536 values at the byte"]
numerators = [d["readout_correct"][k] for k in ("reference","neighbor")] + [d["switches"][k] for k in ("reference","neighbor","whole_site")]
denominators = [d["test_pairs"]]*2 + [d["clean"]]*3
bars = ax.barh(labels,[n/m*100 for n,m in zip(numerators,denominators)],color=["#88dcbc"]*2+["#edbc83"]*3)
ax.bar_label(bars,[f"{n}/{m}" for n,m in zip(numerators,denominators)],padding=6)
ax.invert_yaxis()
ax.set(xlim=(0,120),xlabel="Both letters decoded correctly or both model answers reversed (%)")
ax.spines[["top","right"]].set_visible(False)
fig.savefig(here/"reference-origin.png",dpi=165)
