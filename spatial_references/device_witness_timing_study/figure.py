"""Show estimator latency and the paired ratio for all worlds."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
run=json.loads((root/"audit.json").read_text())
plt.style.use("dark_background")
fig,(a,b)=plt.subplots(1,2,figsize=(8.6,3.3),layout="constrained")
fig.patch.set_facecolor("#101916")
for ax in (a,b):
    ax.set_facecolor("#101916")
    ax.spines[["top","right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x",alpha=.24)
bars=a.barh(["Full NumPy array","Direct MLX gathers"],run["median_ms"],color=("#b6acf0","#84dfc2"),height=.52)
a.bar_label(bars,labels=[f"{x:.2f} ms" for x in run["median_ms"]],padding=4,fontsize=8)
a.set_xlim(0,9.3)
a.set_xlabel("Pooled median estimator latency")
a.invert_yaxis()
ratio=run["world_ratios"]
b.scatter(ratio,range(len(ratio)),color="#84dfc2",s=33)
b.axvline(run["paired_speedup"],ls="--",color="#dfb582",lw=1.25,label="Paired median")
b.set_yticks(range(len(ratio)),[str(x) for x in run["world_ids"]])
b.set_xlabel("Full-array latency ÷ direct-sample latency")
b.set_ylabel("World seed")
b.invert_yaxis()
b.legend(frameon=False,fontsize=8,loc="lower right")
fig.savefig(root/"direct-samples.png",dpi=170)
