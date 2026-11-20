"""Plot the measured change in validation loss when the wide fits use a lower rate."""
from pathlib import Path
import json
import matplotlib.pyplot as plt

here = Path(__file__).resolve().parent
values = json.loads((here / "audit.json").read_text())["performance"]
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(7.8, 3.4), layout="constrained", facecolor="#101916")
ax.set_facecolor("#101916")
for i, (arm, name) in enumerate((("writable", "Writable control"), ("fixed", "Fixed 192 values"))):
    before = values["original_" + arm]["bpb"]
    after = values["recovery_" + arm]["bpb"]
    ax.plot([after, before], [i, i], color="#52675f", lw=2, zorder=1)
    for value, label, color in ((before, "Original learning rate", "#f1bd86"),
                                (after, "One-third learning rate", "#82dbc0")):
        ax.scatter(value, i, s=70, color=color, label=label if i == 0 else None)
        ax.annotate(f"{value:.3f}", (value, i), xytext=(0, 10), textcoords="offset points", ha="center")
ax.axvline(3.24, color="#a2adc4", ls=":", label="Qualification cutoff")
ax.set(yticks=[0,1], yticklabels=["Writable control","Fixed 192 values"], ylim=(-.45,1.5), xlim=(0,6.25),
       xlabel="Validation bits per byte after 4,096 updates (lower is better)")
ax.legend(loc="upper center", bbox_to_anchor=(.5,-.25), ncol=3, frameon=False, fontsize=9)
ax.spines[["top","right","left"]].set_visible(False)
fig.savefig(here / "paired-recovery.png", dpi=165)
