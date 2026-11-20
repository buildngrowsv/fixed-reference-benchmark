from pathlib import Path
import json

import matplotlib.pyplot as plt

here = Path(__file__).resolve().parent
scores = json.loads((here/"audit.json").read_text())["performance"]
plt.rcParams.update({"figure.facecolor":"#101916","axes.facecolor":"#101916","text.color":"#edf4ed",
                     "axes.labelcolor":"#dce7df","xtick.color":"#dce7df","ytick.color":"#dce7df",
                     "axes.spines.top":False,"axes.spines.right":False,"font.size":12})
fig, ax = plt.subplots(figsize=(8.2,4.3),layout="constrained")
names = ["small_fixed","small_writable","large_fixed","large_writable","unigram","bigram"]
labels = ["32M · fixed","32M · writable","285M · fixed","285M · writable","Byte frequency","Previous-byte counts"]
vals = [scores[k]["bpb"] for k in names]
bars = ax.barh(labels,vals,color=["#77d6b5","#9aabbf","#77d6b5","#9aabbf","#c3af84","#c3af84"])
ax.bar_label(bars,fmt="%.3f",padding=6)
ax.invert_yaxis()
ax.set(xlim=(0,6.5),xlabel="Validation bits per byte · lower is better")
ax.xaxis.grid(alpha=.12)
ax.set_axisbelow(True)
fig.savefig(here/"python-width.png",dpi=165)
