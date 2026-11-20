"""Plot word retention with three matched witness policies."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
run=json.loads((root/"audit.json").read_text())
reuse=json.loads((root.parent/"sampled_witness_study/audit.json").read_text())
dense=reuse["dense_routing"]
arms=(("Reuse boundary lines",reuse["rates"]["inferred"],"#dfb582"),
      ("Dedicated detector pairs",run["rates"]["inferred"],"#84dfc2"),
      ("Dense witness statistic",dense,"#b6acf0"))
plt.style.use("dark_background")
fig,ax=plt.subplots(figsize=(7.7,3.2),layout="constrained")
fig.patch.set_facecolor("#101916")
ax.set_facecolor("#101916")
for i,(label,row,color) in enumerate(arms):
    q=100*row["shifted"]["retention"]
    ax.scatter(q,i,color=color,s=68,zorder=3)
    ax.annotate(f"{q:.2f}%",(q,i),xytext=(8,0),textcoords="offset points",va="center",color=color,fontsize=10)
ax.set_yticks(range(3),[r[0] for r in arms])
ax.set_ylim(2.5,-.5)
ax.set_xlim(66.5,72)
ax.set_xlabel("Exact 32-bit words recovered after shifted faults (%)")
ax.set_axisbelow(True)
ax.grid(axis="x",alpha=.25)
ax.tick_params(axis="y",length=0,pad=8)
ax.spines[["top","right","left"]].set_visible(False)
fig.savefig(root/"dedicated-probes.png",dpi=170)
