import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import roc_curve,roc_auc_score,average_precision_score
root=Path('/workspace/Temporal-segmentation-model');out=root/'work/V35_cross_tube_v2';pub=Path('/workspace/shared/downloads');rows=list(csv.DictReader(open(out/'evaluation_metrics.csv')));hist=list(csv.DictReader(open(out/'development/history.csv')));prod=list(csv.DictReader(open(out/'production/history.csv')))
for p in Path('/tmp/v66-fonts').glob('*.TTF'):font_manager.fontManager.addfont(str(p))
plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black','axes.titleweight':'bold','pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white','axes.linewidth':.8})
labels=list(csv.DictReader(open(out/'data/manifest.csv')));test=[r for r in labels if r['split']=='test'];gt=[];pred={name:[] for name in ['V35_original','V35_cross_tube_development']};total={name:np.zeros(4,dtype=np.int64) for name in pred}
for r in test:
 g=np.array(Image.open(out/'data/mask'/r['image_id']));v=g!=128;gt.append(g[v]==255)
 for name,folder in [('V35_original','baseline_predictions'),('V35_cross_tube_development','development_predictions')]:
  p=np.load(out/folder/'cross_tube_test'/(r['image_id']+'_probability.npy'));pp=p[v];gg=g[v]==255;pred[name].append(pp);b=pp>=.5;total[name]+=np.array([(b&gg).sum(),(b&~gg).sum(),(~b&gg).sum(),(~b&~gg).sum()]);m=next(x for x in rows if x['model']==name and x['split']=='cross_tube_test')
for name in pred:assert total[name].tolist()==[int(next(x for x in rows if x['model']==name and x['split']=='cross_tube_test')[k]) for k in ['TP','FP','FN','TN']]
y=np.concatenate(gt);aucrows=[];curves={}
for name,parts in pred.items():
 pp=np.concatenate(parts);fpr,tpr,thr=roc_curve(y,pp);auc=roc_auc_score(y,pp);ap=average_precision_score(y,pp);curves[name]=(fpr,tpr,auc);aucrows.append(dict(model=name,split='cross_tube_test',ROC_AUC=auc,Average_precision=ap,pixels=len(y),root_pixels=int(y.sum())))
with (out/'auc_metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(aucrows[0]));w.writeheader();w.writerows(aucrows)
fig,ax=plt.subplots(2,2,figsize=(7.3,6.1));fig.subplots_adjust(left=.09,right=.98,top=.94,bottom=.13,hspace=.43,wspace=.32)
blue='#2E6F9E';orange='#D68C25';gray='#6A7077'
a=ax[0,0];a.plot([int(r['epoch']) for r in hist[1:]],[float(r['train_optimization_loss']) for r in hist[1:]],color=blue,label='Grouped development');a.plot([int(r['epoch']) for r in prod[1:]],[float(r['train_optimization_loss']) for r in prod[1:]],color=orange,marker='o',markersize=3,label='All-label refit');a.set(xlabel='Fine-tuning epoch',ylabel='Training objective',title='Binary root training');a.legend(frameon=False,fontsize=8)
a=ax[0,1];xx=[int(r['epoch']) for r in hist];a.plot(xx,[float(r['cross_tube_validation_Dice'])*100 for r in hist],color=orange,label='Cross-tube validation');a.plot(xx,[float(r['original_validation_Dice'])*100 for r in hist],color=blue,label='Original validation');best=json.loads((out/'completion.json').read_text())['development_selected_epoch'];a.axvline(best,color=gray,linestyle=':',linewidth=1);a.set(xlabel='Fine-tuning epoch',ylabel='Dice (%)',title='Validation and model selection');a.legend(frameon=False,fontsize=8,loc='center right')
a=ax[1,0];keys=['Precision','Recall','Dice','IoU','Accuracy'];x=np.arange(len(keys));w=.35
for delta,name,color,label in [(-w/2,'V35_original',gray,'Original V35'),(w/2,'V35_cross_tube_development',blue,'Cross-tube V35')]:
 m=next(r for r in rows if r['model']==name and r['split']=='cross_tube_test');vv=[float(m[k])*100 for k in keys];bars=a.bar(x+delta,vv,w,color=color,label=label)
 for b,v in zip(bars,vv):a.text(b.get_x()+b.get_width()/2,v+1,f'{v:.1f}',ha='center',va='bottom',fontsize=7,rotation=90)
a.set(xticks=x,xticklabels=keys,ylim=(0,113),ylabel='Score (%)',title='Held-out cross-tube comparison');a.set_yticks([0,25,50,75,100]);a.tick_params(axis='x',labelsize=8);a.legend(frameon=False,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.14),ncol=2)
a=ax[1,1]
for name,color,label in [('V35_original',gray,'Original V35'),('V35_cross_tube_development',blue,'Cross-tube V35')]:
 fpr,tpr,auc=curves[name];ix=np.unique(np.r_[np.arange(0,len(fpr),max(1,len(fpr)//3000)),len(fpr)-1]);a.plot(fpr[ix],tpr[ix],color=color,lw=1.4,label=f'{label} (AUC = {auc:.4f})')
a.plot([0,1],[0,1],color='#A5A5A5',linestyle='--',lw=.8);a.set(xlim=(0,1),ylim=(0,1.02),xlabel='False positive rate',ylabel='True positive rate',title='Held-out cross-tube ROC');a.legend(frameon=False,fontsize=8,loc='lower right');a.set_aspect('equal',adjustable='box')
for letter,a in zip('ABCD',ax.flat):a.text(-.17,1.08,letter,transform=a.transAxes,fontweight='bold',fontsize=15);a.grid(axis='y',color='#E1E5E8',lw=.45);a.set_axisbelow(True)
for ext in ['pdf','svg','png']:fig.savefig(pub/f'V35_cross_tube_training_evaluation.{ext}',dpi=600 if ext=='png' else 300)
plt.close(fig)
# Native-size qualitative examples: one date per held-out position with the greatest original Dice deficit.
per=list(csv.DictReader(open(out/'metrics/baseline_cross_tube_test_per_image.csv')));ids=[]
for pos in sorted({r['position'] for r in test}):
 candidates=[r for r in test if r['position']==pos];lookup={r['image_id']:r for r in per};ids.append(min(candidates,key=lambda r:float(lookup[r['image_id']]['Dice'] or 1)))
fig,axes=plt.subplots(len(ids),4,figsize=(7.2,len(ids)*1.75));fig.subplots_adjust(left=.02,right=.98,bottom=.01,top=.95,wspace=.025,hspace=.08)
for i,r in enumerate(ids):
 im=np.array(Image.open(out/'data/RGB'/r['image_id']));g=np.array(Image.open(out/'data/mask'/r['image_id']));masks=[None,g==255,np.load(out/'baseline_predictions/cross_tube_test'/(r['image_id']+'_probability.npy'))>=.5,np.load(out/'development_predictions/cross_tube_test'/(r['image_id']+'_probability.npy'))>=.5]
 for j,mask in enumerate(masks):
  arr=im.copy()
  if mask is not None:arr[mask]=(arr[mask]*.25+np.array([255,192,35])*.75).astype('uint8')
  axes[i,j].imshow(arr,interpolation='nearest');axes[i,j].axis('off')
  if i==0:axes[i,j].set_title(['RGB','Uploaded root labels','Original V35','Cross-tube V35'][j],fontsize=9)
 axes[i,0].text(.02,.02,r['image_id'].replace('_ROI500','').replace('.png',''),color='white',fontsize=7,transform=axes[i,0].transAxes,bbox=dict(facecolor='black',alpha=.6,pad=2))
for ext in ['pdf','png']:fig.savefig(pub/f'V35_cross_tube_prediction_examples.{ext}',dpi=400)
plt.close(fig)
print('AUC',aucrows);print('EXAMPLES', [r['image_id'] for r in ids]);print('FIGURES_READY')
