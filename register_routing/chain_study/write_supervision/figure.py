"""Show the same fresh-panel accuracy ranges under soft and discrete routing."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

folder=Path(__file__).resolve().parent
fits=json.loads((folder/"discrete/result.json").read_text())["fits"]
groups=(("donor_clean_soft","Answer training, soft hops","#829eaa"),
        ("donor_write_soft","Write training, soft hops","#e2ba82"),
        ("donor_clean_discrete","Answer training, discrete hops","#7edcb6"),
        ("donor_write_discrete","Write training, discrete hops","#b3e5d0"))

with plt.rc_context({"figure.facecolor":"#101916","axes.facecolor":"#101916","axes.edgecolor":"#50685b",
                     "xtick.color":"#bad1c6","ytick.color":"#e5eee9","text.color":"#e5eee9","font.size":10}):
    fig,axs=plt.subplots(1,3,figsize=(11.8,4.8),sharey=True,layout="constrained")
    for ax,phase,title in zip(axs,("clean","current","stale"),("Clean","Current link edited","Obsolete link edited")):
        for y,(name,label,color) in enumerate(groups):
            samples=[100*f["primary"]["scores"][name][phase]["accuracy"] for f in fits.values()]
            lo,hi=min(samples),max(samples)
            ax.hlines(y,lo,hi,color=color,lw=3)
            ax.scatter([lo,hi],[y,y],color=color,s=15)
            reading=f"{lo:.1f}–{hi:.1f}%" if lo!=hi else f"{lo:.0f}%"
            ax.text(min(hi+.43,101.0),y,reading,va="center",ha="left",color=color)
        ax.set_title(title,loc="left",pad=13)
        ax.axvline(90,color="#688073",lw=1,ls="--")
        ax.set_xlim(85.5,104)
        ax.set_xticks([86,90,95,100])
        ax.spines[["top","right","left"]].set_visible(False)
        ax.tick_params(axis="y",length=0)
        ax.set_xlabel("Accuracy (%)",color="#bad1c6")
    axs[0].set_yticks(range(len(groups)),[g[1] for g in groups])
    axs[0].invert_yaxis()
    fig.suptitle("Discrete choices recovered errors from soft routing",x=.02,ha="left",fontsize=17)
    fig.savefig(folder/"routing-comparison.png",dpi=160)
