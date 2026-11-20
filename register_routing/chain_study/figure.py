"""Plot the four held-out comparisons from the published result."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

folder=Path(__file__).resolve().parent
fits=json.loads((folder/"results/result.json").read_text())["fits"]
names=["convolution-503","convolution-607","recurrent-503","recurrent-607"]
for arm in ("graft","scratch","pretrained"):
    assert all(arm in fits[name]["held_out"]["arms"] for name in names)

plt.rcParams.update({"figure.facecolor":"#101916","axes.facecolor":"#101916",
    "axes.edgecolor":"#50685b","xtick.color":"#b9cec2","ytick.color":"#e5eee9",
    "axes.labelcolor":"#b9cec2","text.color":"#e5eee9","font.size":11})
fig,ax=plt.subplots(figsize=(9.7,4.7),layout="constrained")
row=np.arange(len(names))
for offset,arm,legend,color in ((-.24,"scratch","Native from scratch","#849eaf"),
                                (0,"graft","Frozen transplant","#79dfb4"),
                                (.24,"pretrained","Native with matched pretraining","#e8bc80")):
    rate=[100*fits[n]["held_out"]["arms"][arm]["accuracy"] for n in names]
    ax.barh(row+offset,rate,height=.20,color=color,label=legend)
    for i,v in enumerate(rate):
        ax.text(v+1.1,i+offset,f"{v:.1f}%",va="center",color=color,fontsize=9.5)
ax.set_yticks(row,["Convolution · draw 1","Convolution · draw 2","GRU · draw 1","GRU · draw 2"])
ax.invert_yaxis()
ax.set_xlim(0,100)
ax.spines[["top","right","left"]].set_visible(False)
ax.tick_params(axis="y",length=0)
ax.set_title("One frozen router improves both hosts over scratch",loc="left",pad=16)
ax.set_xlabel("Correct answers on 60-write programs")
ax.legend(loc="lower right",labelcolor="#e5eee9",frameon=False,ncols=1)
fig.savefig(folder/"host-comparison.png",dpi=160)
