"""Plot retention for each routing strategy with matched fault directions."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
data = json.loads((root / "audit.json").read_text())["results"]
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(8.5, 3.9), layout="constrained")
fig.patch.set_facecolor("#101721")
ax.set_facecolor("#101721")
arms = ("fixed", "calibrated", "adaptive", "oracle")
labels = ("Fixed axis", "Calibrate once", "Adapt continuously", "Given orientation")
for offset, env, color, title in ((-.16, "stable", "#75d8eb", "Stable faults"),
                                  (.16, "rotated", "#ffbf80", "Rotated faults")):
    rates = [100 * data[arm][env]["retention"] for arm in arms]
    bars = ax.bar(np.arange(len(arms)) + offset, rates, width=.29, label=title, color=color)
    ax.bar_label(bars, labels=[f"{value:.1f}%" for value in rates], padding=4, fontsize=9)
ax.set_ylim(0, 98)
ax.set_yticks([0, 20, 40, 60, 80], ["0%", "20%", "40%", "60%", "80%"])
ax.set_xticks(range(len(arms)), labels)
ax.set_ylabel("Exact words recovered after faults")
ax.grid(axis="y", color="#45566a", alpha=.45)
ax.spines[["right", "top"]].set_visible(False)
ax.set_axisbelow(True)
ax.legend(loc="upper left", ncol=2, frameon=False)
fig.savefig(root / "adaptive-witness.png", dpi=160)
