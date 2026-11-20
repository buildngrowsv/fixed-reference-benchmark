"""Show the cost of sharing boundary and witness observations."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
d = json.loads((root / "audit.json").read_text())
plt.style.use("dark_background")
fig, (left, right) = plt.subplots(1,2,figsize=(9.4,3.35),layout="constrained")
fig.patch.set_facecolor("#101916")
layouts=("aligned","shifted")
for ax in (left,right):
    ax.set_facecolor("#101916")
    ax.set_yticks([0,1],["Aligned faults","Shifted faults"])
    ax.set_ylim(1.5,-.5)
    ax.spines[["left","right","top"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x",alpha=.25)
for j,(color,name,arm) in enumerate((("#b6acf0","Dense witness",d["dense_routing"]),
                                     ("#84dfc2","Reuse probe lines",d["rates"]["inferred"]))):
    p=[100*arm[x]["retention"] for x in layouts]
    y=[0+(j-.5)*.24,1+(j-.5)*.24]
    left.scatter(p,y,label=name,color=color,s=56,zorder=3)
    for x,i in zip(p,y,strict=True):
        left.annotate(f"{x:.2f}%",(x,i),xytext=(6,0),textcoords="offset points",va="center",color=color,fontsize=9)
left.set_xlim(66,72.3)
left.set_title("Exact word retention",loc="left")
left.set_xlabel("Retention (%)")
left.legend(loc="upper left",bbox_to_anchor=(-.03,-.23),frameon=False,fontsize=8,ncol=2)
for i,layout in enumerate(layouts):
    p=d["paired_minus_full_routing"][layout]
    m=100*p["mean"]; lo,hi=[100*t for t in p["interval"]]
    right.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt="o",capsize=3,color="#84dfc2")
    right.annotate(f"{m:+.2f}",(m,i),xytext=(0,8),textcoords="offset points",ha="center",fontsize=9,color="#84dfc2")
right.axvline(-1,linestyle="--",color="#e7bd85",label="Registered one-point margin")
right.set_title("Difference from dense witness routing",loc="left")
right.set_xlabel("Paired difference (percentage points)")
right.legend(loc="upper left",bbox_to_anchor=(-.03,-.23),frameon=False,fontsize=8)
fig.savefig(root/"sampled-witness.png",dpi=170)
