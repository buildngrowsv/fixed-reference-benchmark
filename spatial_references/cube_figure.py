"""Show the learned sensor locations and a controlled channel substitution."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

R=Path(__file__).resolve().parent/"results"
s=json.loads((R/"summary.json").read_text())["cube"]
weights=np.mean([json.loads((R/f"cube-full-{k}.json").read_text())["gate"] for k in (101,211,307)],axis=0)
fig=plt.figure(figsize=(10.7,4.4))
fig.subplots_adjust(left=.06,right=.97,wspace=.48,top=.82,bottom=.19)
ax=fig.add_subplot(121,projection="3d")
bx=fig.add_subplot(122)
fig.suptitle("Which neighbor tells a voxel to save?",x=.04,y=.97,ha="left",fontsize=18,weight="bold")
for x,y,z in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
    ax.plot([0,x],[0,y],[0,z],color="#a9b4b7",lw=1.6)
    ax.scatter(x,y,z,s=105,color=("#c15d2d" if z==-1 else "#b2c0c3"),depthshade=False)
ax.scatter(0,0,0,s=145,color="#22766d",depthshade=False)
ax.text(0,0,.16,f"Local A  +{weights[0]:.1f}",color="#175e55",fontsize=10)
ax.text(.12,.05,-1,f"Neighbor B  +{weights[6]:.1f}",color="#95431d",fontsize=10)
ax.view_init(elev=18,azim=-50)
ax.set_box_aspect((1,1,1));ax.set_axis_off()
ax.set_title("A   Mean learned gate weights",loc="left",fontsize=11,weight="bold")
keys=("full_full","full_missing","full_irrelevant","missing_missing")
labels=("Intact signals","Relevant B replaced","Irrelevant B replaced","Trained without relevant B")
for i,k in enumerate(keys):
    pct=100*s[k]["exact"]
    bx.barh(i,pct,height=.54,color=("#22766d","#c15d2d","#22766d","#84969b")[i])
    bx.text(pct+2,i,"100%" if pct==100 else f"{pct:.1f}%",va="center")
bx.set(yticks=range(4),yticklabels=labels,xlim=(0,121),xlabel="Exact 32-bit words (%)")
bx.invert_yaxis();bx.set_axisbelow(True);bx.grid(axis="x",alpha=.15)
bx.spines[["top","right","left"]].set_visible(False);bx.tick_params(axis="y",length=0)
bx.set_title("B   Effect on held-out volumes",loc="left",fontsize=11,weight="bold")
fig.text(.035,.045,"Six spatial neighbors. The other six gate inputs averaged small negative weights. Three paired training seeds.",fontsize=9)
fig.savefig(R/"implicit-volume.png",dpi=164,facecolor="white")
