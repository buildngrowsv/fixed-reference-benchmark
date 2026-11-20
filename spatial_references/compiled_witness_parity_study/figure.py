"""Plot observer data counts and recurrent retention parity by world."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
d=json.loads((root/"audit.json").read_text())
plt.style.use("dark_background")
f,(left,right)=plt.subplots(1,2,figsize=(9.6,3.45),layout="constrained")
f.patch.set_facecolor("#101916")
for ax in (left,right):
    ax.set_facecolor("#101916")
    ax.spines[["top","right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(alpha=.24)
labels=("Full field size","Read inside MLX","Returned to NumPy")
vals=(d["cells"],d["sampled_scalars"],d["observer_output_scalars"])
for yer,label,v,color in zip(range(3),labels,vals,("#a3ada5","#b6acf0","#84dfc2"),strict=True):
    left.scatter(v,yer,color=color,s=51,zorder=3)
    left.annotate(f"{v:,}",(v,yer),xytext=(6,0),textcoords="offset points",va="center",color=color,fontsize=8)
left.set_xscale("log")
left.set_xlim(7,8e7)
left.set_yticks(range(3),labels)
left.set_ylim(2.48,-.48)
left.set_xlabel("Scalar values per observation (log scale)")
left.set_title("Compiled observer input and output",loc="left")
x=d["worlds"]
a=[100*v for v in d["reference_retention_by_world"]]
b=[100*v for v in d["shifted_retention_by_world"]]
right.plot(x,a,"o-",color="#b6acf0",ms=4.5,label="Archived NumPy observer")
right.scatter(x,b,facecolors="none",edgecolors="#84dfc2",s=69,lw=1.4,label="Compiled observer")
right.set_xticks(x)
right.set_ylim(min(a)-.28,max(a)+.33)
right.set_title("Shifted-fault retention matches",loc="left")
right.set_xlabel("World")
right.set_ylabel("Exact 32-bit recovery (%)")
right.legend(frameon=False,loc="upper center",bbox_to_anchor=(.5,-.2),ncol=2,fontsize=7.5)
f.savefig(root/"compiled-parity.png",dpi=170)
