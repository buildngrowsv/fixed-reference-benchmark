import json
from pathlib import Path
import matplotlib.pyplot as plt

here=Path(__file__).resolve().parent
v=json.loads((here/"audit.json").read_text())["balanced"]
plt.rcParams.update({"figure.facecolor":"#101916","axes.facecolor":"#101916","text.color":"#eef5ef","axes.labelcolor":"#dbe8df",
                     "xtick.color":"#dbe8df","ytick.color":"#dbe8df","font.size":12,"axes.spines.top":False,"axes.spines.right":False})
fig,ax=plt.subplots(figsize=(8,3.9),layout="constrained")
names=["Ordinary · reduced rate","Routed · reduced rate","Routed · original rate"]
bars=ax.barh(names,[100*v[k] for k in ("quarter_dense","quarter_rotated","original_rotated")],color=["#8d9fb9","#76d3b3","#6fae9a"])
ax.axvline(100/32,color="#eac682",ls="--",label="Chance · 3.125%")
ax.bar_label(bars,fmt="%.2f%%",padding=4)
ax.invert_yaxis()
ax.set(xlim=(0,6),xlabel="Balanced recall after 4,096 updates (%)")
ax.legend(loc="lower right",frameon=False)
ax.xaxis.grid(alpha=.12);ax.set_axisbelow(True)
fig.savefig(here/"long-recall.png",dpi=165)
