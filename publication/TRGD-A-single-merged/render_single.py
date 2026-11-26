"""Combine new/existing root metric panels into one shared-axis scientific plot."""
import csv,json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
ROOT=Path('/workspace/Temporal-segmentation-model');HERE=Path(__file__).resolve().parent
for p in Path('/tmp/v66-fonts').glob('*.TTF'):font_manager.fontManager.addfont(str(p))
plt.rcParams.update({'font.family':'Times New Roman','font.size':15,'axes.labelsize':17,'axes.labelweight':'bold','xtick.labelsize':17,'ytick.labelsize':15,'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black','pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
def read(p):return list(csv.DictReader(open(p,encoding='utf-8-sig')))
latest=read(ROOT/'work/TRGD_latest_whole_ablation/metrics.csv');old=read(ROOT/'work/V90_trgd_evaluation/metrics.csv');image=[next(r for r in old if r['model']=='V66 U-Net'),next(r for r in old if r['model']=='TRGD on')]
for version,rows in [('latest',latest),('attached_values',image)]:
 folder=HERE/version;folder.mkdir(parents=True,exist_ok=True);fig,ax=plt.subplots(figsize=(10.6,5.8));fig.subplots_adjust(left=.11,right=.95,bottom=.22,top=.73);x=np.arange(4);ks=['Precision','Recall','Dice','IoU'];colors={'new_root':'#D89500','existing_root':'#2D73A6'};plots=[];data=[]
 for cls in ['new_root','existing_root']:
  name='New root' if cls=='new_root' else 'Existing root'
  for i,row in enumerate(rows):
   vals=np.array([float(row[cls+'_'+k])*100 for k in ks]);base='V66 prior (TRGD off)' if version=='latest' else 'V66 U-Net';model=base if i==0 else 'TRGD (V90)'
   line,=ax.plot(x,vals,ls=(0,(4,2)) if i==0 else '-',marker='o' if i==0 else 's',ms=7.5,lw=1.7,color=colors[cls],markerfacecolor='white' if i==0 else colors[cls],markeredgewidth=1.5,label=f'{name} · {model}');plots.append(line)
   dx,dy,ha={('existing_root',0):(-15,17,'right'),('existing_root',1):(15,3,'left'),('new_root',0):(-15,-19,'right'),('new_root',1):(15,-12,'left')}[cls,i]
   for xx,y,k in zip(x,vals,ks):
    ax.annotate(f'{y:.2f}',(xx,y),xytext=(dx,dy),textcoords='offset points',ha=ha,va='center',fontsize=13,fontweight='bold' if k=='Dice' else 'normal',color='black')
    data.append(dict(root_class=name,model=model,metric=k,score_percent=float(y)))
 ax.set_xlim(-.38,3.48);ax.set_ylim(0,113);ax.set_xticks(x,ks);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Score (%)');ax.tick_params(axis='x',length=0,pad=9);ax.grid(axis='y',alpha=.13,lw=.7);ax.set_axisbelow(True)
 fig.text(.5,.985,'New and existing root classification',ha='center',va='top',fontsize=20,fontweight='bold')
 # Order columns by model, rows by root class.
 order=[0,2,1,3];fig.legend([plots[i] for i in order],[plots[i].get_label() for i in order],loc='upper center',bbox_to_anchor=(.53,.91),ncol=2,frameon=False,fontsize=12.5,columnspacing=2,handlelength=2.8)
 macro=[float(r['macro_Dice'])*100 for r in rows];fig.text(.53,.065,f'Macro Dice  {macro[0]:.2f}% → {macro[1]:.2f}%',ha='center',fontsize=18,fontweight='bold')
 prefix='TRGD_A_merged_single'
 for ext,dpi in [('pdf',350),('svg',350),('png',450)]:fig.savefig(folder/(prefix+'.'+ext),dpi=dpi,facecolor='white')
 fig.savefig(folder/(prefix+'.tiff'),dpi=600,facecolor='white',pil_kwargs={'compression':'tiff_lzw'});fig.savefig(folder/'preview.png',dpi=160,facecolor='white');plt.close(fig)
 with (folder/'plot_values.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 proof=dict(version=version,source='work/TRGD_latest_whole_ablation/metrics.csv' if version=='latest' else 'work/V90_trgd_evaluation/metrics.csv; rows V66 U-Net and TRGD on',macro_Dice_percent=macro,rows=rows,figure='One shared Cartesian axis; four model/class series; no qualitative image grid',class_colors={'new_root':colors['new_root'],'existing_root':colors['existing_root']},fonts='Black Times New Roman')
 (folder/'provenance.json').write_text(json.dumps(proof,indent=2)+'\n');print('SINGLE_PLOT_READY',version,macro,flush=True)
