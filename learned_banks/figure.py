import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

R=Path(__file__).resolve().parent/"results"
s=json.loads((R/"summary.json").read_text())["conditions"]
fig,(ax,bx)=plt.subplots(1,2,figsize=(10.7,4.7))
fig.subplots_adjust(left=.085,right=.97,top=.78,bottom=.22,wspace=.39)
fig.suptitle("Can two interchangeable banks learn different jobs?",x=.035,y=.96,ha="left",fontsize=18,weight="bold")
palette={"assigned":"#236b65","neutral":"#b45e32","exchange":"#566f8a"}
names={"assigned":"Assigned roles","neutral":"Learned roles","exchange":"Midpoint exchange"}
for name,c in palette.items():
    ax.plot([0,25,100,250,500],100*np.array(s[name]["mean_curve"]),"o-",color=c,label=names[name],lw=1.8,ms=4)
ax.set(xlabel="Training updates",ylabel="Correct 32-bit words (%)",ylim=(-4,105))
ax.legend(loc="center right",frameon=False,fontsize=9)
ax.set_title("A   Learning on validation cubes",loc="left",fontsize=11,weight="bold")
for i,name in enumerate(palette):
    value=s[name]["metrics"]["primary"]["exact"]*100
    bx.bar(i,value,color=palette[name],width=.62)
    bx.text(i,value+2,f"{value:.1f}%",ha="center",fontsize=11)
bx.set(xticks=range(3),xticklabels=["Assigned\nroles","Learned\nroles","Midpoint\nexchange"],ylim=(0,110),ylabel="Correct 32-bit words (%)")
bx.set_title("B   Larger, longer held-out cubes",loc="left",fontsize=11,weight="bold")
for axis in (ax,bx):
    axis.spines[["top","right"]].set_visible(False);axis.grid(axis="y",alpha=.15);axis.set_axisbelow(True)
fig.text(.035,.035,"Both banks hold 32 values per voxel. Lines and bars show the mean of three paired seeds.",fontsize=9)
fig.savefig(R/"learning-bank-roles.png",dpi=165,facecolor="white")
