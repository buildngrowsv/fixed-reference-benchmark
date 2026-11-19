import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
R=Path(__file__).resolve().parent/"results"
s=json.loads((R/"summary.json").read_text())
fig,(ax,bx)=plt.subplots(1,2,figsize=(11.1,4.7))
fig.subplots_adjust(left=.16,right=.98,wspace=.83,bottom=.18,top=.77)
fig.suptitle("When should the network rewrite its reference?",x=.023,y=.96,ha="left",fontsize=18,weight="bold")
def panel(axis,labels,numbers,title):
    for i,(name,num) in enumerate(zip(labels,numbers)):
        axis.barh(i,num*100,height=.5,color=("#2a5c73","#bc6237","#267564","#628a81")[i])
        txt="100" if num==1 else ("0" if num==0 else (f"{100*num:.2f}" if num<.01 else f"{100*num:.1f}"))
        axis.text(100*num+2,i,txt,va="center",fontsize=10)
    axis.set(yticks=np.arange(len(labels)),yticklabels=labels,xlim=(0,116),xlabel="32-bit words correct (%)")
    axis.invert_yaxis()
    axis.spines[["top","right","left"]].set_visible(False)
    axis.tick_params(axis="y",length=0)
    axis.grid(axis="x",alpha=.16);axis.set_axisbelow(True)
    axis.set_title(title,loc="left",fontsize=11,weight="bold")
panel(ax,["Fixed reference","Signed write","Writes disabled¹","Bounded write"],[s["fresh"][k]["exact"] for k in ("fixed","writable","writes_off","convex")],"A   Restore the original source")
panel(bx,["Fixed initial reference","Event-writable","SAVE writes disabled²"],[s["snapshot"]["24"][k]["exact"] for k in ("fixed_normal","event_normal","event_no_save")],"B   Restore the latest snapshot")
fig.text(.03,.025,"¹ Same signed-write network.     ² Same event-writable network.     Both panels use depth 24 and 1,024 fresh boards.",fontsize=9)
fig.savefig(R/"spatial-reference.png",dpi=168,facecolor="white")
