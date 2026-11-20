"""Plot same-round gains with evolved and common saved-word histories."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
data = json.loads((root / "audit.json").read_text())
values = np.column_stack((np.asarray(data["gains_by_jump"]), data["gains"])) * 100
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(7.9, 4.2), facecolor="#101916", layout="constrained")
ax.set_facecolor("#101916")
x = np.arange(4)
for i, name, color in ((0, "Separate saved-word histories", "#b6acf0"),
                       (1, "Common saved-word bank at each jump", "#84dfc2")):
    ax.bar(x + (i-.5)*.32, values[i], .29, label=name, color=color)
ax.axhline(0, lw=.8, color="#c4d1cc")
ax.set(title="The pooled gain held up when the saved words started equal",
       ylabel="Sparse observer's gain over stale calibration (points)", ylim=(-2.2, 6.8))
ax.set_xticks(x, ["First jump", "Second jump", "Third jump", "Pooled"])
ax.legend(frameon=False)
ax.grid(axis="y", alpha=.18)
fig.savefig(root / "common-gain.png", dpi=170)
