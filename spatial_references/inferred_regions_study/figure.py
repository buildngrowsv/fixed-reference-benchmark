"""Show memory behavior after translating the partition away from its default location."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
rates = json.loads((root / "audit.json").read_text())["rates"]
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(7.9, 4.05), layout="constrained")
fig.patch.set_facecolor("#101721")
ax.set_facecolor("#101721")
arms = ("fixed", "inferred", "known")
labels = ("Fixed boundary", "Inferred boundary", "Known boundary")
for j, (arm, label, color) in enumerate(zip(arms, labels, ("#fac18a", "#70d8ea", "#b9d79c"), strict=True)):
    values = [100 * rates[arm][environment]["retention"] for environment in ("aligned", "shifted")]
    bars = ax.bar(np.arange(2) + (j-1)*.23, values, width=.21, label=label, color=color)
    ax.bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=4)
ax.set_xlim(-.5, 1.5)
ax.set_ylim(0, 100)
ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
ax.set_xticks([0,1], ["Original partition", "Translated partition"])
ax.set_ylabel("Exact words recovered")
ax.grid(axis="y", color="#495f73", alpha=.45)
ax.set_axisbelow(True)
ax.spines[["right","top"]].set_visible(False)
ax.legend(loc="upper center", ncol=3, frameon=False)
fig.savefig(root / "inferred-regions.png", dpi=160)
