import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

here = Path(__file__).resolve().parent
scores = json.loads((here / "audit.json").read_text())["balanced"]
plt.rcParams.update({"figure.facecolor":"#101916","axes.facecolor":"#101916","text.color":"#eef4ee",
                     "axes.labelcolor":"#d4dfda","xtick.color":"#d4dfda","ytick.color":"#d4dfda",
                     "font.size":12,"axes.spines.top":False,"axes.spines.right":False,"axes.spines.left":False})
fig, ax = plt.subplots(figsize=(8,4.2), layout="constrained")
x = np.arange(2)
for shift, arm, label, color in [(-.19,"dense","Ordinary attention","#8da2bd"),(.19,"rotated","One routed head","#70d6b4")]:
    values = [100*scores[f"{size}_{arm}"] for size in ("small","large")]
    bars = ax.bar(x+shift, values, .33, color=color, label=label,zorder=2)
    ax.bar_label(bars, fmt="%.2f%%", padding=4, color="#eef4ee",fontsize=11)
ax.axhline(100/32,color="#e7c582",lw=1.3,ls="--",label="Chance · 3.125%")
ax.set(xticks=x,xticklabels=["14.3 million weights","227 million weights"],ylim=(0,105),ylabel="Balanced recall (%)")
ax.legend(loc="upper right",frameon=False)
ax.set_axisbelow(True)
ax.yaxis.grid(color="#314138",alpha=.6)
fig.savefig(here/"recall.png",dpi=165)
