"""Draw the two retention comparisons for a dark reading page."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

here = Path(__file__).resolve().parent
scores = json.loads((here / "audit.json").read_text())
background = "#101721"
text = "#edf4ff"
grid = "#3e4c60"
colors = ("#ff947d", "#f1d283", "#72dcdd")

plt.rcParams.update({"figure.facecolor": background, "axes.facecolor": background,
                     "text.color": text, "axes.labelcolor": text, "xtick.color": text,
                     "ytick.color": text, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})
fig, (left, right) = plt.subplots(1, 2, figsize=(9.8, 4.4), sharey=True)
for ax, title in zip((left, right), ("One write, longer wait", "Repeated writes, matched faults"), strict=True):
    ax.set_title(title, loc="left", pad=13, fontweight="bold")
    ax.set_ylim(0, 113)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", alpha=.45, color=grid, linewidth=.7)
    ax.set_axisbelow(True)
    ax.spines[["left", "bottom"]].set_color(grid)
    ax.set_xticks(range(2))
left.set_ylabel("Exact 32-bit recovery")
left.set_xticklabels(["Step 16", "Step 128"])
right.set_xticklabels(["Separated pulses", "Consecutive pulses"])

rates = scores["long_hold"]["rates"]
for offset, prefix, label, color in ((-.15, "soft", "Soft gate", colors[0]),
                                      (.15, "hard", "Thresholded gate", colors[2])):
    bars = left.bar(np.arange(2) + offset, [100 * rates[f"{prefix}_{x}"] for x in ("short", "long")],
                    width=.26, label=label, color=color)
    left.bar_label(bars, labels=([">99.99", "0"] if prefix == "soft" else ["100", "100"]), padding=4, fontsize=9)
for offset, policy, label, color in zip((-.24, 0, .24),
        ("single_hard", "double_soft", "double_hard"), ("One tick", "Two soft ticks", "Two hard ticks"), colors, strict=True):
    values = [100 * scores["timed_faults"]["rates"][s][policy]["after_fault_interval"]
              for s in ("separated", "bundled")]
    bars = right.bar(np.arange(2) + offset, values, width=.22, label=label, color=color)
    right.bar_label(bars, labels=[f"{v:.1f}" if v < 100 else "100" for v in values], padding=4, fontsize=9)
for ax in (left, right):
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.15), frameon=False, ncol=3, fontsize=9)

fig.subplots_adjust(left=.095, right=.98, top=.91, bottom=.25, wspace=.26)
fig.savefig(here / "retention.png", dpi=160)
