"""Plot shifted-world memory retention and boundary localization."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
audit = json.loads((root / "audit.json").read_text())
modes = ("random", "stratified", "full")
labels = ("Random 64", "Stratified 64", "Full observation")
colors = ("#ddad75", "#84dfc2", "#b6acf0")
shifted = audit["comparators"]
retention = [100*shifted[m]["rates"]["shifted"]["retention"] for m in modes]
accuracy = [100*shifted[m]["boundary"]["shifted"]["exact"] for m in modes]
plt.style.use("dark_background")
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.25), layout="constrained", sharey=True)
fig.patch.set_facecolor("#101916")
for ax, values, title, limits in zip(axes, (retention, accuracy),
                                    ("Words recovered", "Exact boundary coordinates"),
                                    ((64, 75), (60, 97)), strict=True):
    ax.set_facecolor("#101916")
    ax.set_title(title, loc="left", pad=11, fontweight="semibold")
    ax.set_yticks(range(3), labels)
    ax.set_xlim(*limits)
    ax.set_ylim(2.6, -.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=.24)
    ax.set_xlabel("Rate under shifted faults (%)")
    for i, (value, color) in enumerate(zip(values, colors, strict=True)):
        ax.scatter(value, i, s=72, color=color, zorder=3)
        ax.annotate(f"{value:.2f}%", (value, i), xytext=(7, 0), textcoords="offset points",
                    ha="left", va="center", color=color, fontsize=9)
fixed = 100*audit["rates"]["fixed"]["shifted"]["retention"]
axes[0].axvline(fixed, color="#a7b5b0", ls="--", lw=1.3)
axes[0].text(fixed + .18, 2.41, "Fixed boundary", color="#a7b5b0", va="bottom", fontsize=8)
fig.savefig(root / "probe-placement.png", dpi=170)
