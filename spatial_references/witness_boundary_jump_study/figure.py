"""Plot the shifted condition's per-round exact retention."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
with np.load(root / "results/jump.npz") as raw:
    counts = raw["counts"][0, :, :, :, 1].sum(axis=0)
rate = 100 * counts[..., 3] / counts[..., 2]
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(8.2, 4.2), layout="constrained", facecolor="#101916")
ax.set_facecolor("#101916")
for j, label, color, style in ((2, "Supplied boundaries", "#b6acf0", "--"),
                               (0, "Stale calibration", "#e8bc79", "-"),
                               (1, "Sparse inference", "#84dfc2", "-")):
    ax.plot(np.arange(1, 17), rate[:, j], style, color=color, lw=2, label=label)
ax.axvline(8.5, color="#91a19a", lw=1, ls=":")
ax.set(title="Retention after the boundary shift", ylabel="Exact words recovered (%)", xlabel="Round")
ax.set_xticks([1, 4, 8, 9, 12, 16])
ax.grid(axis="y", alpha=.18)
ax.legend(frameon=False, loc="lower left")
fig.savefig(root / "trajectory.png", dpi=170)
