"""Draw the planned two contrasts at the compact and full widths."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

here = Path(__file__).resolve().parent
d = json.loads((here/"audit.json").read_text())["test"]
plt.style.use("dark_background")
fig,ax = plt.subplots(figsize=(8.6,3.3),layout="constrained",facecolor="#101916")
ax.set_facecolor("#101916")
names = ("sealed_minus_readable","sealed_minus_working")
labels = ("238M sealed vs 285M readable","238M sealed vs 238M late copy")
for i,key in enumerate(names):
    row = d["contrasts"][key]
    p = row["delta"]
    lo,hi = row["interval"]
    ax.errorbar(p,i,xerr=[[p-lo],[hi-p]],fmt="o",color="#86dbbb",capsize=5)
    ax.annotate(f"{p:+.4f}",(hi,i),xytext=(7,0),textcoords="offset points",va="center")
ax.axvline(0,color="#a3aec2",ls="--",lw=1)
ax.set(yticks=range(2),yticklabels=labels,ylim=(-.42,1.42),xlim=(-.028,.012),
       xlabel="Held-out loss difference in bits per byte (negative favors sealed)")
ax.invert_yaxis()
ax.spines[["top","right"]].set_visible(False)
fig.savefig(here/"compact-held-out.png",dpi=165)
