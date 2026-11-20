"""Plot paired original and external domain losses before and after adaptation."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

here = Path(__file__).resolve().parent
d = json.loads((here/"audit.json").read_text())["arms"]
plt.style.use("dark_background")
fig,axes = plt.subplots(1,2,figsize=(8.5,3.5),layout="constrained",facecolor="#101916")
for ax,(domain,title,limits) in zip(axes,[("cpython","Original CPython",(1.065,1.098)),("external","Six external projects",(1.24,1.79))]):
    ax.set_facecolor("#101916")
    for k,label,c in (("sealed","Sealed early copy","#86dbbb"),("working","Late copy","#a9bad7")):
        vals = [d[k][domain+"_"+phase] for phase in ("before","after")]
        ax.plot([0,1],vals,marker="o",label=label,color=c)
    ax.set(xticks=[0,1],xticklabels=["Before","After"],xlim=(-.16,1.16),ylim=limits,title=title)
    ax.spines[["top","right"]].set_visible(False)
axes[0].set_ylabel("Held-out bits per byte (lower is better)")
axes[1].legend(frameon=False,loc="upper right",fontsize=8)
fig.savefig(here/"retention.png",dpi=165)
