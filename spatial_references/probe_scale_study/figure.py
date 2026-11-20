"""Show retained words and the paired full-observation margin at each grid size."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
small = json.loads((root.parent / "stratified_regions_study/audit.json").read_text())
large = json.loads((root / "audit.json").read_text())
labels = ("884,736 cells", "7,077,888 cells")
fixed = [small["rates"]["fixed"]["shifted"]["retention"],
         large["modes"]["stratified"]["rates"]["fixed"]["shifted"]["retention"]]
full = [small["comparators"]["full"]["rates"]["shifted"]["retention"],
        large["modes"]["full"]["rates"]["inferred"]["shifted"]["retention"]]
sampled = [small["rates"]["inferred"]["shifted"]["retention"],
           large["modes"]["stratified"]["rates"]["inferred"]["shifted"]["retention"]]
contrasts = (small["sparse_minus_full"]["shifted"], large["stratified_minus_full"]["shifted"])
colors = ("#a3aca5", "#b6acf0", "#84dfc2")
plt.style.use("dark_background")
fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 3.65), layout="constrained")
fig.patch.set_facecolor("#101916")
for ax in (left, right):
    ax.set_facecolor("#101916")
    ax.set_yticks([0, 1], labels)
    ax.set_ylim(1.48, -.51)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.grid(axis="x", alpha=.24)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
for column, color, name, j in zip((fixed, full, sampled), colors,
                                  ("Fixed boundaries", "Full observation", "64 probes"), (-1, 0, 1), strict=True):
    rate = [100*v for v in column]
    positions = [0+j*.15, 1+j*.15]
    left.scatter(rate, positions, color=color, s=48, label=name, zorder=3)
    for x, y in zip(rate, positions, strict=True):
        left.annotate(f"{x:.2f}%", (x, y), xytext=(6, 0), textcoords="offset points",
                      color=color, fontsize=8, va="center")
left.set_xlim(100*min(*fixed, *full, *sampled)-1.1, 100*max(*fixed, *full, *sampled)+2.65)
left.set_title("Exact word retention", loc="left")
left.set_xlabel("Shifted-fault retention (%)")
left.legend(loc="upper left", bbox_to_anchor=(-.01, -.25), ncol=3, frameon=False, fontsize=8)
right.axvline(0, color="#687d73", lw=1)
right.axvline(-2, color="#e7bd85", linestyle="--", lw=1.4, label="Registered −2-point margin")
for y, item in enumerate(contrasts):
    point = 100*item["mean"]
    lo, hi = [100*value for value in item["interval"]]
    right.errorbar(point, y, xerr=[[point-lo],[hi-point]], fmt="o", color=colors[2], markersize=6, capsize=3)
    right.annotate(f"{point:+.2f}", (point, y), xytext=(0, 8), textcoords="offset points",
                   color=colors[2], fontsize=9, ha="center")
right.set_title("Difference from full observation", loc="left")
right.set_xlabel("Paired difference (percentage points)")
right.legend(loc="upper left", bbox_to_anchor=(-.01, -.25), frameon=False, fontsize=8)
fig.savefig(root / "scale.png", dpi=170)
