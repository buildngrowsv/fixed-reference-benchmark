from pathlib import Path
import json
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.facecolor":"#101916","axes.facecolor":"#101916","text.color":"#edf1ef","axes.labelcolor":"#edf1ef","xtick.color":"#d8dfdb","ytick.color":"#d8dfdb","font.size":12,"axes.spines.top":False,"axes.spines.right":False})
folder=Path(__file__).resolve().parent
audit=json.loads((folder/'audit.json').read_text())['specific_answer_swaps']
fig,ax=plt.subplots(figsize=(7,3.7),layout='constrained')
for i,k in enumerate(('compact','full')):
 a=audit[k]
 ax.barh(i,a['clean'],color='#637289',height=.45)
 ax.barh(i,a['source'],color='#73d6bc',height=.45)
 ax.text(50,i,f"{a['source']:,} source swaps  /  {a['clean']:,} initially correct",color='#10261e',fontweight='bold',va='center')
ax.set(yticks=[0,1],yticklabels=['Compact donor','Full branch'],xlim=(0,2048),xlabel='Program pairs (2,048 tested)',title='The later-write control switched zero pairs in both donors')
ax.invert_yaxis();fig.savefig(folder/'portability.png',dpi=165)
