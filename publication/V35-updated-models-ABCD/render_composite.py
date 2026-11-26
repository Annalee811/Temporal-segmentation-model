"""Publication figure from measured results: original A–D panel arrangement."""
import csv,json,math,hashlib,shutil,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,ticker
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from PIL import Image
HERE=Path(__file__).resolve().parent;ROOT=Path('/workspace/Temporal-segmentation-model');RUN=ROOT/'work/V35_full250_all66';PUB=Path('/workspace/shared/downloads');PREFIX='V35_updated_models_ABCD'
for f in Path('/tmp/v66-fonts').glob('*.TTF'):font_manager.fontManager.addfont(str(f))
plt.rcParams.update({'font.family':'Times New Roman','font.size':14,'axes.titlesize':21,'axes.titleweight':'bold','axes.labelsize':16,'axes.labelweight':'bold','xtick.labelsize':13,'ytick.labelsize':13,'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8,'legend.frameon':False})
def readcsv(p):return list(csv.DictReader(open(p,encoding='utf-8-sig')))
def readjson(p):return json.loads(p.read_text())
models=['unet','deeplab','segformer'];names=['U-Net','DeepLabV3+','SegFormer-B0'];linecolors=['#1676A8','#D68027','#248D85'];blue='#46779A';gold='#BD9549';red='#A55C65';gray='#81878C'
results={m:readjson(HERE/(m+'_complete.json')) for m in models};records={m:readcsv(HERE/(m+'_parent_metrics.csv')) for m in models}
assert all(r['monitoring_roc_pixels']==3145728 for r in results.values())
assert all(r['monitoring_root_pixels']==153297 for r in results.values())
assert len({tuple(r['image_id'] for r in records[m]) for m in models})==1
fig=plt.figure(figsize=(15.8,16.8),facecolor='white');outer=fig.add_gridspec(4,1,height_ratios=[3.0,7.3,.25,3.35],left=.065,right=.985,top=.94,bottom=.085,hspace=.40)
top=outer[0].subgridspec(1,4,wspace=.38,width_ratios=[1,1,1,1.28])
lossaxes=[]
for i,m in enumerate(models):
 ax=fig.add_subplot(top[i]);lossaxes.append(ax)
 if m=='unet':
  h=[r for r in readcsv(RUN/'history.csv') if int(r['epoch'])>0];tk='train_loss';vl='validation_loss';labels=['Weighted train loss','Monitoring loss'];subtitle='Updated V35 | selected epoch 91'
 else:
  h=readcsv(ROOT/'02_root_background/assets/weights'/m/'training_curves.csv');tk='train_optimization_loss';vl='validation_loss';labels=['Training loss','Validation loss'];subtitle='Retained baseline | archived training'
 ep=np.array([int(r['epoch']) for r in h]);assert np.array_equal(ep,np.arange(1,101))
 ax.plot(ep,[float(r[tk]) for r in h],color='#177BA0',lw=1.5,label=labels[0]);ax.plot(ep,[float(r[vl]) for r in h],color='#D78426',lw=1.5,label=labels[1])
 ax.set_title(names[i],pad=25);ax.text(.5,1.025,subtitle,transform=ax.transAxes,ha='center',fontsize=12.5);ax.set_xlabel('Epoch');ax.set_ylabel('Loss');ax.set_xlim(1,100);ax.set_xticks([1,25,50,75,100]);ax.legend(fontsize=11.5,loc='upper right');ax.grid(alpha=.15);ax.set_axisbelow(True)
 if m=='unet':ax.axvline(91,color='#838383',lw=.7,ls=':',alpha=.8)
rocax=fig.add_subplot(top[3])
for m,n,col in zip(models,names,linecolors):
 roc=np.load(HERE/(m+'_roc.npz'));rocax.plot(roc['fpr'],roc['tpr'],color=col,lw=1.65,ls='-.' if m=='segformer' else '-',label=f"{n} (AUC = {results[m]['monitoring_auc']:.4f})")
rocax.plot([0,1],[0,1],color='#969696',lw=1,ls=(0,(4,3)),label='Chance');rocax.set(xlim=(-.01,1.01),ylim=(-.01,1.015),xlabel='False positive rate',ylabel='True positive rate');rocax.set_title('ROC comparison',pad=25);rocax.text(.5,1.025,'Same 48 monitoring parent images',transform=rocax.transAxes,ha='center',fontsize=12.5);rocax.legend(loc='lower right',fontsize=12.5);rocax.grid(alpha=.15);rocax.xaxis.set_major_locator(ticker.MultipleLocator(.25));rocax.yaxis.set_major_locator(ticker.MultipleLocator(.25))
# C: same parent grids, three source groups, no invented third evaluation partition.
center=outer[1].subgridspec(1,3,wspace=.25)
maxlen=max(max(float(r[k]) for m in models for r in records[m]) for k in ['reference_length_px','predicted_length_px']);lim=math.ceil(maxlen/1000)*1000;lim=max(4000,lim)
allres=np.array([float(r['predicted_length_px'])-float(r['reference_length_px']) for m in models for r in records[m]]);rlo=math.floor(min(-500,allres.min())/500)*500;rhi=math.ceil(max(500,allres.max())/500)*500
scatteraxes=[]
for i,m in enumerate(models):
 grid=center[i].subgridspec(3,2,width_ratios=[1,.15],height_ratios=[.9,4.5,1.5],wspace=.035,hspace=.07)
 topax=fig.add_subplot(grid[0,0]);ax=fig.add_subplot(grid[1,0]);scatteraxes.append(ax);right=fig.add_subplot(grid[1,1],sharey=ax);res=fig.add_subplot(grid[2,0],sharex=ax)
 rows=records[m];x=np.array([float(r['reference_length_px']) for r in rows]);y=np.array([float(r['predicted_length_px']) for r in rows]);split=np.array([r['split'] for r in rows]);source=np.array([r['source'] for r in rows]);groups=[((split=='train')&(source=='original_244'),blue,'o'),((split=='train')&(source=='submitted_66'),gold,'^'),(split=='val',red,'o')]
 for use,col,marker in groups:
  ax.scatter(x[use],y[use],s=21 if col==red else 16,color=col,alpha=.67,edgecolors='white',linewidths=.25,marker=marker,zorder=3)
  res.scatter(x[use],(y-x)[use],s=20 if col==red else 15,color=col,alpha=.67,edgecolors='white',linewidths=.2,marker=marker)
  bins=np.linspace(0,lim,15);topax.hist(x[use],bins=bins,color=col,alpha=.18,density=True);right.hist(y[use],bins=bins,color=col,alpha=.18,density=True,orientation='horizontal')
  if np.std(x[use])>0:
   t=np.linspace(0,lim,200);topax.plot(t,gaussian_kde(x[use])(t),color=col,lw=1)
  if np.std(y[use])>0:
   t=np.linspace(0,lim,200);right.plot(gaussian_kde(y[use])(t),t,color=col,lw=1)
 coef=results[m]['regression']['train'];xx=np.array([0,lim]);ax.plot(xx,coef['slope']*xx+coef['intercept'],color='#262626',lw=1.25,label='Training OLS fit');ax.plot([0,lim],[0,lim],color='#737373',ls=(0,(4,3)),lw=1.1,label='Identity (y = x)')
 stats=[]
 for label,key in [('Train','train'),('Monitor','val')]:
  z=results[m]['regression'][key];stats.extend([f"{label}: R² = {z['R2']:.3f} | OLS R² = {z['OLS_R2']:.3f}",f"RMSE = {z['RMSE']:.1f} px; MAE = {z['MAE']:.1f} px"])
 ax.text(.035,.975,'\n'.join(stats),transform=ax.transAxes,va='top',fontsize=12.5,fontweight='bold',bbox={'facecolor':'none','edgecolor':'none','pad':0})
 ax.set(xlim=(0,lim),ylim=(0,lim));ax.grid(alpha=.12,ls=':');ax.tick_params(labelbottom=False);ax.xaxis.set_major_locator(ticker.MaxNLocator(4));ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
 if i==0:ax.set_ylabel('Predicted root length (px)')
 res.axhline(0,color='#575757',lw=1);res.set(ylim=(rlo,rhi),xlabel='Reference root length (px)');res.grid(alpha=.14,ls=':');res.yaxis.set_major_locator(ticker.MaxNLocator(4));res.xaxis.set_major_locator(ticker.MaxNLocator(4));res.tick_params(labelsize=12)
 if i==0:res.set_ylabel('Residual (px)',fontsize=15)
 topax.set_xlim(0,lim);topax.set_title(names[i],pad=9);topax.axis('off');right.axis('off')
legend=[Line2D([],[],color=blue,marker='o',ls='',markersize=6,label='Training: existing (196)'),Line2D([],[],color=gold,marker='^',ls='',markersize=6,label='Training: added (66)'),Line2D([],[],color=red,marker='o',ls='',markersize=6,label='Internal monitoring (48)'),Line2D([],[],color='#262626',lw=1.4,label='Training OLS fit'),Line2D([],[],color='#737373',ls='--',label='Identity (y = x)')]
# position in the gap between center and bottom rows
bottom=outer[3].subgridspec(1,3,wspace=.55)
confaxes=[]
for i,m in enumerate(models):
 ax=fig.add_subplot(bottom[i]);confaxes.append(ax);z=results[m]['counts']['val'];counts=np.array([[z['TN'],z['FP']],[z['FN'],z['TP']]]);norm=counts/counts.sum(1,keepdims=True)
 ax.imshow(norm,vmin=0,vmax=1,cmap='Blues',alpha=.66,interpolation='nearest')
 for r in range(2):
  for col in range(2):
   ax.text(col,r-.11,f'{norm[r,col]*100:.1f}%',ha='center',va='center',fontsize=18,fontweight='bold',color='black')
   ax.text(col,r+.12,f'n = {counts[r,col]:,}',ha='center',va='center',fontsize=14,fontweight='bold',color='black')
 ax.set_xticks([0,1],['Background','Root']);ax.set_yticks([0,1],['Background','Root']);ax.set_xlabel('Prediction');ax.set_ylabel('Reference');ax.set_title(names[i],pad=15);ax.tick_params(length=0);ax.set_aspect('equal')
 # captions stay outside the matrix to preserve the original row-normalized layout.
 ax.text(.5,-.23,f"Dice = {z['Dice']*100:.2f}% | IoU = {z['IoU']*100:.2f}%",transform=ax.transAxes,ha='center',fontsize=12.5)
fig.canvas.draw()
# Panel labels placed against actual axes geometry.
for letter,ax in [('A',lossaxes[0]),('B',rocax),('C',scatteraxes[0]),('D',confaxes[0])]:
 box=ax.get_position();y=box.y1+.012
 if letter=='C':y=outer[1].get_position(fig).y1+.02
 if letter=='D':y=outer[3].get_position(fig).y1+.015
 fig.text(.015 if letter!='B' else box.x0-.05,y,letter,fontsize=30,fontweight='bold',va='bottom')
cy=outer[1].get_position(fig).y0;dy=outer[3].get_position(fig).y1
fig.legend(handles=legend,loc='center',bbox_to_anchor=(.535,(cy+dy)/2),ncol=5,fontsize=12.5,columnspacing=1.4,handletextpad=.6)
import os
if os.environ.get('FIGURE_PREVIEW_ONLY')=='1':
 fig.savefig(HERE/'preview.png',dpi=140,facecolor='white');raise SystemExit(0)
for ext,dpi in [('pdf',350),('svg',350),('png',400)]:
 fig.savefig(PUB/(PREFIX+'.'+ext),dpi=dpi,facecolor='white');print('SAVED',ext,flush=True)
fig.savefig(HERE/'preview.png',dpi=140,facecolor='white')
fig.savefig(PUB/(PREFIX+'.tiff'),dpi=600,facecolor='white',pil_kwargs={'compression':'tiff_lzw'});plt.close(fig);print('SAVED tiff',flush=True)
caption='''Figure. Updated root–background segmentation and root-length agreement. (A) Actual 100-epoch loss histories. Updated V35 U-Net was continued on 1,048 native250 training crops (784 existing-data and264 additional-data crops); its selected checkpoint is epoch91. DeepLabV3+ and SegFormer-B0 are retained baseline checkpoints, whose archived training histories used the earlier244-file dataset (200training/20validation/24legacy-test image files). Their histories are not new310-file training runs. The V35 weighted training objective and unweighted monitoring objective have different weighting and are not directly comparable in absolute scale. (B) Pixel-level ROC curves recomputed on the identical48 current monitoring parent images (3,145,728pixels;153,297root pixels), using native250 windows, stride188 and Hann probability blending for every model. (C) Measured versus predicted skeleton lengths on the same310 parent grids:196existing-data training files,66additional training files and48monitoring files. Length is the number of skeleton pixels after Zhang skeletonization, without pruning, spatial resizing or physical-length conversion. Parent sizes are256×256(existing) and500×500(additional). The panels include marginal distributions, training-only ordinary least-squares fits, identity lines and prediction-minus-reference residuals. R² denotes1−SSE/SST against the identity target; OLS R² measures a separate fitted linear relation within each partition; the displayed OLS line uses training parents only. (D) Row-normalized confusion matrices at the fixed0.5probability threshold for the same48monitoring parents. Each parent grid is scored once without repeatedly scoring its overlapping inference windows; parent images and temporal observations remain correlated. Spatially overlapping250px child views are not independent samples. The current dataset has no independent test split. Of the48monitoring files,44had prior gradient-training exposure through inherited V35 weights. Baselines received different historical training data; these panels describe current development performance, not a controlled retraining benchmark or independent generalization. All RGB, reference masks and archived model weights were preserved.
'''
# Normalize spaces for the readable caption.
import re
caption=re.sub(r'(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])',' ',caption).replace('V 35','V35').replace('V 3','V3').replace('B 0','B0').replace('OLSR','OLS R')
(PUB/(PREFIX+'_caption.txt')).write_text(caption)
provenance=dict(figure=PREFIX,models={m:results[m] for m in models},datasets=dict(training_crops=1048,monitoring_crops=192,training_parents=262,monitoring_parents=48,independent_test=0),training_curves=dict(unet='current100epochs',deeplab='archived100epochs',segformer='archived100epochs'),deeplab_equivalence=readjson(HERE/'deeplab_equivalence.json'),font='Times New Roman',text_color='black')
(PUB/(PREFIX+'_provenance.json')).write_text(json.dumps(provenance,indent=2)+'\n')
for m in models:
 for tail in ['_parent_metrics.csv','_complete.json','_roc.npz']:shutil.copy2(HERE/(m+tail),PUB/(PREFIX+'_'+m+tail))
readme='''# Updated V35 and retained baseline models: A–D scientific figure\n\nPDF/SVG are vector output; PNG is400dpi; TIFF is600dpi with lossless LZW compression. Latin text uses embedded Times New Roman in PDF, black text throughout. The complete caption explains the actual evaluation sets and historical training differences.\n\nU-Net: latest V35 selected epoch91, after100epochs. DeepLabV3+ and SegFormer-B0: preserved baseline checkpoints, not newly retrained. B/C/D use new model inference on matched native images, native250 sliding windows and the same current parent partition. The24legacy-test files were used only to verify the DeepLab inference translation; they are not used as a new independent test set.\n\nThe current48parent monitoring files correspond to192correlated training-sized views, with each parent pixel counted once in B/D. The264additional crops all belong to training. Root-length scatter uses262training parents and48monitoring parents. No fabricated third partition is shown. Historical weight exposure is recorded in the caption.\n'''
(PUB/(PREFIX+'_README.md')).write_text(readme)
# Package figure, caption, machine-readable measurements and exact plotting/inference sources.
with zipfile.ZipFile(PUB/(PREFIX+'_bundle.zip'),'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f in sorted(PUB.glob(PREFIX+'*')):
  if f.suffix!='.zip' and not f.name.endswith('_checksums.json'):z.write(f,f.name)
 for f in ['render_composite.py','evaluate_updated.py','deeplab_npz_forward.py','deeplab_equivalence.json']:z.write(HERE/f,'code/'+f)
checks={f.name:dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(PUB.glob(PREFIX+'*')) if not f.name.endswith('_checksums.json')}
(PUB/(PREFIX+'_checksums.json')).write_text(json.dumps(checks,indent=2)+'\n');print('FIGURE_COMPLETE',flush=True)
