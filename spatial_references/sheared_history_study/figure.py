"""Plot paired retained-word percentages and exact offset recovery."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

here=Path(__file__).resolve().parent
data=json.loads((here/"audit.json").read_text())
plt.style.use("dark_background")
fig,ax=plt.subplots(1,2,figsize=(9.3,4.3),layout="constrained",facecolor="#101916")
for a in ax:
    a.set_facecolor("#101916")
    a.invert_yaxis()
    a.grid(axis="x",alpha=.18)
colors=["#b6acf0","#84dfc2","#e8bc79"]
r=ax[0].barh(["Current frame","Pooled frames","Supplied geometry"],[100*x[1] for x in data["retention"]],color=colors)
ax[0].set(title="Saved-word retention",xlabel="Exact words (%)",xlim=(0,85))
ax[0].bar_label(r,fmt="%.2f",padding=4)
r=ax[1].barh(["Current frame","Pooled frames"],[26,data["recovered_offsets"]],color=colors[:2])
ax[1].set(title="Boundary positions",xlabel="Exact offset vectors out of 64",xlim=(0,77))
ax[1].bar_label(r,padding=4)
fig.savefig(here/"pooled.png",dpi=170)
