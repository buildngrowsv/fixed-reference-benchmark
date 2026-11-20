"""Plot the scale comparison with the paired dense-orientation margin."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
small=json.loads((root.parent/"anchored_witness_study/audit.json").read_text())
small_dense=json.loads((root.parent/"stratified_regions_study/audit.json").read_text())
large=json.loads((root/"audit.json").read_text())
arms=(("Fixed geometry",[small["rates"]["fixed"],large["rates"]["fixed"]],"#9daaa2"),
      ("Both decisions sampled",[small["rates"]["inferred"],large["rates"]["inferred"]],"#84dfc2"),
      ("Dense witness",[small_dense["rates"]["inferred"],large["dense_orientation"]],"#b6acf0"))
names=["884,736 cells\n5.79% field","7,077,888 cells\n0.984% field"]
plt.style.use("dark_background")
fig,(left,right)=plt.subplots(1,2,figsize=(10.4,3.75),layout="constrained")
fig.patch.set_facecolor("#101916")
for ax in (left,right):
    ax.set_facecolor("#101916")
    ax.set_yticks(range(2),names)
    ax.set_ylim(1.46,-.46)
    ax.spines[["top","left","right"]].set_visible(False)
    ax.tick_params(axis="y",length=0)
    ax.set_axisbelow(True)
    ax.grid(axis="x",alpha=.23)
for i,(label,data,color) in enumerate(arms):
    v=[100*x["shifted"]["retention"] for x in data]
    y=[j+(i-1)*.16 for j in range(2)]
    left.scatter(v,y,label=label,color=color,s=49,zorder=3)
    for x,p in zip(v,y,strict=True):
        left.annotate(f"{x:.2f}%",(x,p),xytext=(6,0),textcoords="offset points",va="center",fontsize=8,color=color)
left.set_title("Shifted-fault word retention",loc="left")
left.set_xlim(62,73.7)
left.set_xlabel("Exact words recovered (%)")
left.legend(loc="upper left",bbox_to_anchor=(-.025,-.23),ncol=2,frameon=False,fontsize=8)
for i,row in enumerate((small,large)):
    v=row["paired_minus_dense"]["shifted"]
    m=100*v["mean"]
    lo,hi=[100*x for x in v["interval"]]
    right.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt="o",capsize=3,color="#84dfc2")
    right.annotate(f"{m:+.2f}",(m,i),xytext=(0,9),textcoords="offset points",ha="center",fontsize=9,color="#84dfc2")
right.axvline(-1,ls="--",color="#e0b782",label="Registered one-point margin")
right.axvline(0,color="#7b8c83",lw=1)
right.set_title("Difference from dense witness",loc="left")
right.set_xlabel("Paired difference (percentage points)")
right.set_xlim(-1.08,.08)
right.legend(loc="upper left",bbox_to_anchor=(-.025,-.23),frameon=False,fontsize=8)
fig.savefig(root/"sub-one-percent.png",dpi=170)
