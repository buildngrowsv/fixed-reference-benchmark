"""Plot post-fault exact word recovery on the larger regional grid."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
data = json.loads((root / "audit.json").read_text())["rates"]
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(7.7, 4.15), layout="constrained")
fig.patch.set_facecolor("#101721")
ax.set_facecolor("#101721")
arms = ("global", "regional", "orientation")
names = ("One global direction", "A direction per region", "Given orientation")
colors = ("#f3b991", "#72d9ed", "#afd4b4")
for j, (arm, title, color) in enumerate(zip(arms, names, colors, strict=True)):
    scores = [100 * data[arm][env]["retention"] for env in ("homogeneous", "heterogeneous")]
    bars = ax.bar(np.arange(2) + (j - 1) * .235, scores, width=.21, label=title, color=color)
    ax.bar_label(bars, labels=[f"{score:.1f}%" for score in scores], padding=4)
ax.set_ylim(0, 102)
ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
ax.set_xticks([0, 1], ["Regions share fault direction", "Regional fault directions disagree"])
ax.set_ylabel("Exact 32-bit word recovery")
ax.grid(axis="y", color="#445466", alpha=.45)
ax.spines[["top", "right"]].set_visible(False)
ax.set_axisbelow(True)
ax.legend(loc="upper center", bbox_to_anchor=(.5, -.13), ncol=3, frameon=False, fontsize=9)
fig.savefig(root / "regional-witness.png", dpi=160)
