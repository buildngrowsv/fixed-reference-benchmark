from pathlib import Path
import json
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.facecolor":"#101916","axes.facecolor":"#101916","text.color":"#edf1ef","axes.labelcolor":"#edf1ef","xtick.color":"#d8dfdb","ytick.color":"#d8dfdb","font.size":12,"axes.spines.top":False,"axes.spines.right":False})
folder=Path(__file__).resolve().parent
audit=json.loads((folder/'audit.json').read_text())
fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
for label,key,color in [('External projects','external_bpb','#73d6bc'),('CPython','cpython_bpb','#efae72')]:
 values=audit[key]; ax.plot(range(3),values,'o-',color=color,lw=2.5,ms=8,label=label)
 for i,v in enumerate(values): ax.annotate(f'{v:.3f}',(i,v),xytext=(0,10),textcoords='offset points',ha='center',color=color)
ax.set(xticks=range(3),xticklabels=['Before adaptation','Ordinary adaptation','1 in 8 CPython batches'],ylabel='Bits per byte · lower is better',ylim=(.95,1.69),xlim=(-.22,2.22))
ax.yaxis.grid(alpha=.13);ax.legend(frameon=False,loc='upper right');fig.savefig(folder/'replay.png',dpi=165);plt.close(fig)
