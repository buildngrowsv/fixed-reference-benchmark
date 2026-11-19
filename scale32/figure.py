"""Render the two measured endpoints from the saved summaries."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent/"results"
edit=json.loads((p/"summary.json").read_text())["results"]
memory=json.loads((p/"bottleneck/summary.json").read_text())["results"]
order=["fixed4","writable32","fixed32"];colors=["#6b7882","#527aa2","#147d67"]
fig,ax=plt.subplots(1,2,figsize=(10,3.4),layout="constrained")
for a,title,values,base in [(ax[0],"Editing after a restore",[edit[k]["test"]["accuracy"] for k in order],50),
                             (ax[1],"Recall after a state reset",[memory[k]["old"] for k in order],12.5)]:
    a.barh([2,1,0],[100*x for x in values],color=colors,height=.55)
    for y,v in zip([2,1,0],values):
        a.text(100*v+1,y,f"{100*v:.2f}%",va="center",fontsize=10)
    a.set_yticks([2,1,0],["4 fixed","32 writable","32 fixed"]);a.set_xlim(0,119)
    a.set_xticks([0,25,50,75,100]);a.set_xlabel("Correct answers (%)");a.set_title(title,loc="left",weight="bold")
    a.axvline(base,ls=":",color="#414141",lw=1.1)
    a.spines[["top","right","left"]].set_visible(False)
fig.suptitle("Reference width changed recall much more than editing",fontsize=16,weight="bold")
fig.savefig(p/"reference-width.png",dpi=165)
