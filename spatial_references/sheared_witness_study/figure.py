"""Plot word recovery and localization under a tilted partition."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

here=Path(__file__).resolve().parent
d=json.loads((here/"audit.json").read_text())
plt.style.use("dark_background")
fig,axs=plt.subplots(1,2,figsize=(9.4,4.1),layout="constrained",facecolor="#101916")
for ax in axs:
    ax.set_facecolor("#101916")
    ax.grid(axis="x",alpha=.18)
    ax.invert_yaxis()
b=axs[0].barh(["Axis-aligned estimate","Inferred tilt and offsets","Exact tilt and offsets"],
             [100*d["retention"][k]["shifted"]["retention"] for k in ("fixed","inferred","known")],
             color=["#b6acf0","#84dfc2","#e8bc79"])
axs[0].set(title="Retention",xlabel="Exact saved words (%)",xlim=(0,90))
axs[0].bar_label(b,fmt="%.2f",padding=4)
b=axs[1].barh(["Correct tilt","All offsets exact"],[d["shifted_correct_shears"],d["shifted_exact_offset_vectors"]],
             color=["#84dfc2","#e8bc79"])
axs[1].set(title="Geometry recovery",xlabel="Correct observations out of 64",xlim=(0,75))
axs[1].bar_label(b,padding=4)
fig.savefig(here/"tilt.png",dpi=170)
