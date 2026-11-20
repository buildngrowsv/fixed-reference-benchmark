"""Plot the fixed-witness ablation in dark mode."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
rates = json.loads((root / "audit.json").read_text())["rates"]
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(7.15, 4), layout="constrained")
fig.patch.set_facecolor("#101721")
ax.set_facecolor("#101721")
for j, (arm, label, color) in enumerate(zip(("two_ticks", "near", "far"),
        ("Two ticks", "One voxel away", "Eight voxels away"), ("#ff947d", "#f1d283", "#72dcdd"), strict=True)):
    values = [rates[arm][fault]["after_faults"] for fault in ("tiles", "columns")]
    bars = ax.bar(np.arange(2) + (j - 1) * .23, np.array(values) * 100, width=.2, label=label, color=color)
    ax.bar_label(bars, labels=[f"{x:.1%}" for x in values], padding=4)
ax.set_ylim(0, 85)
ax.set_yticks([0, 20, 40, 60, 80], ["0%", "20%", "40%", "60%", "80%"])
ax.set_ylabel("Exact 32-bit recovery after faults")
ax.set_xticks([0, 1], ["Compact 4×4×4 faults", "Columns span separation axis"])
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=.4, color="#3e4c60")
ax.spines[["right", "top"]].set_visible(False)
ax.legend(loc="upper right", frameon=False, ncol=1)
fig.savefig(root / "witness.png", dpi=160)
