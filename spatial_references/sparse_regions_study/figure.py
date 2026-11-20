"""Contrast sampled and full inputs to the boundary estimator."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
s = json.loads((root / "audit.json").read_text())
plt.style.use("dark_background")
fig,(left,right)=plt.subplots(1,2,figsize=(8.3,3.8),layout="constrained")
fig.patch.set_facecolor("#101721")
for ax in (left,right):
    ax.set_facecolor("#101721")
    ax.set_ylim(0,110)
    ax.set_yticks([0,25,50,75,100],["0%","25%","50%","75%","100%"])
    ax.set_axisbelow(True)
    ax.grid(axis="y",alpha=.4,color="#495b6e")
    ax.spines[["top","right"]].set_visible(False)

colors=("#fac18a","#73d8ea")
labels=("Limited","Full")
left.set_title("Exact boundary coordinates")
b=left.bar(labels,[100*s["boundary_accuracy"]["shifted"]["exact"],
                   100*s["full_boundary_accuracy"]["shifted"]["exact"]],color=colors,width=.57)
left.bar_label(b,fmt="%.1f%%",padding=4)
right.set_title("Exact words after shifted faults")
b=right.bar(("Fixed",*labels),[100*s["rates"]["fixed"]["shifted"]["retention"],
         100*s["rates"]["inferred"]["shifted"]["retention"],100*s["full_observation"]["shifted"]["retention"]],
         color=("#aeb9c7",*colors),width=.57)
right.bar_label(b,fmt="%.1f%%",padding=4)
fig.savefig(root/"sparse-regions.png",dpi=160)
