"""Plot held-out loss effects and identifier sensitivity."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

here = Path(__file__).resolve().parent
loss = json.loads((here/"audit.json").read_text())["test"]["effects"]
exchange = json.loads((here/"exchange-audit.json").read_text())
plt.style.use("dark_background")
fig, (left,right) = plt.subplots(1,2,figsize=(10,3.7),layout="constrained",facecolor="#101916")
for ax in (left,right):
    ax.set_facecolor("#101916")
    ax.spines[["top","right"]].set_visible(False)
sizes = ("small","large")
labels = ("32M","285M")
left.axvline(0,color="#879ba8",lw=1,ls="--")
for y,k in enumerate(sizes):
    point = loss[k]["delta"]
    lo,hi = loss[k]["interval"]
    left.errorbar(point,y,xerr=[[point-lo],[hi-point]],fmt="o",color="#89d9bf",capsize=4)
left.set(yticks=[0,1],yticklabels=labels,xlabel="Fixed minus writable (bits per byte)",title="Held-out loss · lower is better")
for offset, mode, label, color in ((-.16,"designated","Fixed copy","#89d9bf"),(.16,"neighbor","Working neighbors","#ebbb84")):
    for y,k in enumerate(sizes):
        count = exchange[k]["fixed"]["outcomes"][mode]["lost"]
        bar = right.barh(y+offset,count,height=.3,color=color,label=label if y==0 else None)
        right.bar_label(bar,padding=4)
right.set(yticks=[0,1],yticklabels=labels,xlim=(0,131),xlabel="Previously correct pairs lost",title="Swapping equal-width slices")
right.legend(loc="upper right",frameon=False,fontsize=8)
for ax in (left,right):
    ax.invert_yaxis()
fig.savefig(here/"wide-held-out.png",dpi=165)
