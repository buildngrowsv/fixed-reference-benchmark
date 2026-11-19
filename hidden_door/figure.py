"""Show the difference between ordinary accuracy and intervention-sensitive recovery."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent/"results"
data=json.loads((R/"audit.json").read_text())
fig,axes=plt.subplots(1,2,figsize=(11.4,4.2),gridspec_kw={"width_ratios":[1.2,1]},layout="constrained")
names=("Reads both ends","Blinded to far latch","Unrestricted")
arms=("both_ends","one_end","unrestricted")
ax=axes[0];y=np.arange(3)
for j,(measure,color,legend) in enumerate((("overall","#a3b1bd","All words"),("both","#237a6e","Both settings, affected words"))):
    vals=[]
    for a in arms:
        records=[s["test"][a] for s in data["scores"].values()]
        vals.append(100*np.mean([(r["all_open"]+r["all_sealed"])/(2*r["all_cells"]) if measure=="overall" else r["both"]/r["dependent"] for r in records]))
    bars=ax.barh(y+(-.17 if j==0 else .17),vals,height=.31,color=color,label=legend)
    ax.bar_label(bars,labels=[f"{v:.1f}%" for v in vals],padding=3,fontsize=8)
ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlim(0,118);ax.set_xlabel("Exact 32-bit words (%)");ax.set_title("Open and sealed versions of each cube",loc="left",fontsize=12,fontweight="bold")
ax.spines[["top","right","left"]].set_visible(False);ax.tick_params(axis="y",length=0);ax.legend(loc="lower left",bbox_to_anchor=(0,-.4),fontsize=9,frameon=False)
ax=axes[1]
for i,(arm,color) in enumerate((("both_ends","#237a6e"),("one_end","#a3b1bd"))):
    for j,s in enumerate((101,211,307)):
        z=data["truth_tables"][f"{arm}_{s}"]; p=np.sort(z["row_mass"])
        ax.plot(np.arange(1,65),p,color=color,alpha=.7,label=(("Both ends" if i==0 else "Far latch hidden") if j==0 else None))
ax.axhline(.5,ls=":",c="#666",lw=1);ax.set_ylim(0,1.05);ax.set_xlabel("Local sensor settings, sorted within each fit");ax.set_ylabel("Weight on the correct move");ax.set_title("All 64 possible sensor settings",loc="left",fontsize=12,fontweight="bold")
ax.spines[["top","right"]].set_visible(False);ax.legend(loc="lower right",fontsize=9,frameon=False)
fig.savefig(R/"doorway.png",dpi=170,bbox_inches="tight")

