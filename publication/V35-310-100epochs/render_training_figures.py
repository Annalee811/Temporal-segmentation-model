import csv,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import roc_curve,roc_auc_score,average_precision_score
root=Path('/workspace/Temporal-segmentation-model');out=root/'work/V35_310_100epochs';pub=Path('/workspace/shared/downloads');done=json.load(open(out/'completion.json'));assert done['epochs']==100
h=list(csv.DictReader(open(out/'history.csv')));assert [int(r['epoch']) for r in h]==list(range(101));r=list(csv.DictReader(open(out/'data/manifest.csv',encoding='utf-8-sig')));validation=[x for x in r if x['split']=='val'];metrics=list(csv.DictReader(open(out/'final_validation_metrics.csv')));baseline=list(csv.DictReader(open(out/'baseline_validation_metrics.csv')))[0]
for f in Path('/tmp/v66-fonts').glob('*.TTF'):font_manager.fontManager.addfont(str(f))
plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black','axes.titleweight':'bold','pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white','axes.linewidth':.8})
y=[];probs={label:[] for label in ['Original V35','Best validation','Epoch 100']};conf={label:np.zeros(4,dtype=np.int64) for label in probs}
folders={'Original V35':out/'baseline_validation_predictions','Best validation':out/'validation_predictions/best_validation','Epoch 100':out/'validation_predictions/epoch100'}
for a in validation:
 g=np.array(Image.open(out/'data/mask'/a['image_id']));v=g!=128;gg=g[v]==255;y.append(gg)
 for label,folder in folders.items():
  p=np.load(folder/(a['image_id']+'_probability.npy'));pp=p[v];probs[label].append(pp);b=pp>=.5;conf[label]+=np.array([(b&gg).sum(),(b&~gg).sum(),(~b&gg).sum(),(~b&~gg).sum()])
reference={'Original V35':baseline,'Best validation':metrics[0],'Epoch 100':metrics[1]}
for label,cc in conf.items():assert list(cc)==[int(reference[label][k]) for k in ['TP','FP','FN','TN']]
y=np.concatenate(y);curves={};auc=[]
for label,parts in probs.items():
 p=np.concatenate(parts);fpr,tpr,_=roc_curve(y,p);area=roc_auc_score(y,p);ap=average_precision_score(y,p);curves[label]=(fpr,tpr,area);auc.append(dict(model=label,split='internal_validation',ROC_AUC=area,Average_precision=ap,valid_pixels=len(y),root_pixels=int(y.sum()),independent_validation=False))
with (out/'validation_auc.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(auc[0]));w.writeheader();w.writerows(auc)
blue='#2E6F9E';orange='#D78B28';gray='#68717A';green='#4C8C77';fig,ax=plt.subplots(2,2,figsize=(7.4,6.2));fig.subplots_adjust(left=.09,right=.98,top=.94,bottom=.14,hspace=.46,wspace=.32)
a=ax[0,0];ep=np.array([int(x['epoch']) for x in h[1:]]);a.plot(ep,[float(x['train_loss']) for x in h[1:]],color=blue,lw=1.2,label='Training');a.plot([int(x['epoch']) for x in h],[float(x['validation_loss']) for x in h],color=orange,lw=1.2,label='Validation');a.set(xlabel='Epoch',ylabel='V35 composite objective',title='Root/background training');a.legend(frameon=False,fontsize=8)
a=ax[0,1];a.plot(ep,[float(x['train_Dice'])*100 for x in h[1:]],color=blue,lw=1.2,label='Training crops');a.plot([int(x['epoch']) for x in h],[float(x['validation_Dice'])*100 for x in h],color=orange,lw=1.2,label='Internal validation');a.axvline(done['selected_epoch'],color=gray,ls=':',lw=1);a.set(xlabel='Epoch',ylabel='Dice (%)',title='Segmentation learning curve');a.legend(frameon=False,fontsize=8,loc='best')
a=ax[1,0];keys=['Precision','Recall','Dice','IoU','Accuracy'];x=np.arange(5);w=.25
for offset,label,color in [(-w,'Original V35',gray),(0,'Best validation',blue),(w,'Epoch 100',green)]:
 mm=reference[label];vv=[float(mm[k])*100 for k in keys];bars=a.bar(x+offset,vv,w,color=color,label=label)
 for b,v in zip(bars,vv):a.text(b.get_x()+b.get_width()/2,v+1,f'{v:.1f}',ha='center',va='bottom',rotation=90,fontsize=6)
a.set(xticks=x,xticklabels=keys,ylim=(0,113),yticks=[0,25,50,75,100],ylabel='Score (%)',title='Internal validation metrics');a.tick_params(axis='x',labelsize=8);a.legend(frameon=False,fontsize=7.5,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.15))
a=ax[1,1]
for label,color in [('Original V35',gray),('Best validation',blue),('Epoch 100',green)]:
 fpr,tpr,area=curves[label];ix=np.unique(np.r_[np.arange(0,len(fpr),max(1,len(fpr)//3000)),len(fpr)-1]);a.plot(fpr[ix],tpr[ix],color=color,lw=1.2,label=f'{label} (AUC = {area:.4f})')
a.plot([0,1],[0,1],color='#AAAAAA',ls='--',lw=.8);a.set(xlim=(0,1),ylim=(0,1.02),xlabel='False positive rate',ylabel='True positive rate',title='Internal validation ROC');a.set_aspect('equal',adjustable='box');a.legend(frameon=False,fontsize=7.5,loc='lower right')
for letter,a in zip('ABCD',ax.flat):a.text(-.17,1.08,letter,transform=a.transAxes,fontweight='bold',fontsize=15);a.grid(axis='y',color='#E1E5E8',lw=.45);a.set_axisbelow(True)
for ext in ['pdf','svg','png']:fig.savefig(pub/f'V35_310_100epochs_training.{ext}',dpi=600)
plt.close(fig)
# Show every submitted validation position at its middle available date.
new=[a for a in validation if a['source']=='submitted_66'];selected=[]
for pos in sorted({a['position'] for a in new}):
 ss=sorted([a for a in new if a['position']==pos],key=lambda a:a['period']);selected.append(ss[len(ss)//2])
fig,axes=plt.subplots(len(selected),4,figsize=(7.2,len(selected)*1.8),squeeze=False);fig.subplots_adjust(left=.02,right=.98,top=.94,bottom=.01,wspace=.025,hspace=.08)
for i,a in enumerate(selected):
 im=np.array(Image.open(out/'data/RGB'/a['image_id']));g=np.array(Image.open(out/'data/mask'/a['image_id']));masks=[None,g==255,np.load(folders['Best validation']/(a['image_id']+'_probability.npy'))>=.5,np.load(folders['Epoch 100']/(a['image_id']+'_probability.npy'))>=.5]
 for j,mask in enumerate(masks):
  arr=im.copy()
  if mask is not None:arr[mask]=(arr[mask]*.25+np.array([255,192,35])*.75).astype('uint8')
  axes[i,j].imshow(arr,interpolation='nearest');axes[i,j].axis('off')
  if i==0:axes[i,j].set_title(['RGB','Root labels','Best validation','Epoch 100'][j],fontsize=9)
 axes[i,0].text(.02,.02,a['image_id'].replace('_ROI500','').replace('.png',''),color='white',fontsize=7,transform=axes[i,0].transAxes,bbox=dict(facecolor='black',alpha=.6,pad=2))
for ext in ['pdf','png']:fig.savefig(pub/f'V35_310_100epochs_examples.{ext}',dpi=400)
print('CURVES_AND_METRICS_READY',auc)
