"""Compare pilot validation and final exact-word recovery."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
R=Path(__file__).resolve().parent/"results"
data=json.loads((R/"summary.json").read_text())["arms"]
order=["neutral","persistent","biased","assigned"]
names=["Neutral banks · 468","Symmetric retention · 468","Assigned start · 468","Assigned roles · 73"]
fig,ax=plt.subplots(figsize=(9,3.8),layout="constrained")
y=np.arange(4)
controls=[("#a6b4c1","Validation",lambda r:r["validation"]),("#287e73","Held-out test",lambda r:r["metrics_denominator_then_hits"]["primary"][1]/r["metrics_denominator_then_hits"]["primary"][0])]
for j,(color,name,ratio) in enumerate(controls):
    vals=[100*ratio(data[k]) for k in order]
    bars=ax.barh(y+(.17 if j else -.17),vals,height=.31,color=color,label=name)
    ax.bar_label(bars,labels=[f"{v:.2f}%" if v<100 else "100%" for v in vals],fontsize=8,padding=3)
ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlim(0,111);ax.set_xlabel("Exact recovery of a 32-bit word (%)")
ax.set_title("Does a symmetric retention bias teach two banks different roles?",loc="left",weight="bold",pad=12)
ax.spines[["top","right","left"]].set_visible(False);ax.tick_params(axis="y",length=0);ax.legend(loc="center right",bbox_to_anchor=(1,.75),frameon=False)
fig.savefig(R/"memory.png",dpi=170,bbox_inches="tight")

