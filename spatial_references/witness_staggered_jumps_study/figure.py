"""Render the three paired first-round endpoints in dark mode."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
by_event = np.asarray(json.loads((root / "audit.json").read_text())["by_jump"]) * 100
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(7.6, 4.3), layout="constrained", facecolor="#101916")
ax.set_facecolor("#101916")
for i, label, color, marker in ((0, "Stale calibration", "#e8bc79", "o"),
                                (1, "Sparse inference", "#84dfc2", "s"),
                                (2, "Supplied boundaries", "#b6acf0", "^")):
    ax.plot([1, 2, 3], by_event[:, i], label=label, color=color, marker=marker, lw=2)
ax.set(title="Word retention in the first round after each jump",
       ylabel="Exact words recovered (%)", xlabel="Boundary jump", xlim=(.8, 3.2))
ax.set_xticks([1, 2, 3])
ax.grid(axis="y", alpha=.18)
ax.legend(frameon=False)
fig.savefig(root / "staggered.png", dpi=170)
