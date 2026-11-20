"""Plot paired gains from the two replay interventions."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

here = Path(__file__).resolve().parent
data = json.loads((here / "audit.json").read_text())["comparisons"]
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(8, 3.5), layout="constrained", facecolor="#101916")
ax.set_facecolor("#101916")
for arm, label, color, offset in (("early", "Early copy", "#87dec0", .13),
                                   ("late", "Current-state copy", "#acbfff", -.13)):
    for row, domain in enumerate(("cpython", "external")):
        estimate = data[arm][domain]
        gain = -estimate["delta"]
        lo, hi = estimate["interval"]
        ax.errorbar(gain, row + offset, xerr=[[gain + hi], [-lo - gain]], fmt="o",
                    color=color, capsize=4, markersize=7, label=label if row == 0 else None)
ax.set(yticks=(0, 1), yticklabels=("CPython", "Six external projects"), ylim=(-.48, 1.48),
       xlim=(0, .23), xlabel="Reduction in prediction loss with replay (bits per byte)")
ax.legend(loc="upper right", frameon=False)
ax.grid(axis="x", alpha=.17)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(here / "replay-control.png", dpi=160)
