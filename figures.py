"""Replot scientific comparisons from arrays and numerical curves, without a UI."""
import argparse,csv
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate import ROOT,MODELS,measure
PAL=np.array([[255,255,255],[136,86,167],[0,158,115],[230,171,2],[0,114,178]],np.uint8)
def colors(a):
 b=np.full((*a.shape,3),255,np.uint8);use=np.isin(a,[1,2,3,4]);b[use]=PAL[a[use]];return b
def save(fig,folder,name):
 fig.savefig(folder/(name+'.pdf'),dpi=300);fig.savefig(folder/(name+'.tif'),dpi=600,pil_kwargs={'compression':'tiff_adobe_deflate'});plt.close(fig)
def main():
 p=argparse.ArgumentParser();p.add_argument('--v66',type=Path,required=True);p.add_argument('--v90',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.titleweight':'bold','axes.labelweight':'bold','pdf.fonttype':42})
 f,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
 for ax,(m,name) in zip(axes,MODELS.items()):
  rows=list(csv.DictReader((ROOT/'03_temporal_V66/assets/weights'/m/'training_curves.csv').open(encoding='utf-8-sig')))
  ep=[int(r['epoch']) for r in rows];assert ep==list(range(1,101))
  ax.plot(ep,[float(r['training_loss']) for r in rows],label='Training');ax.plot(ep,[float(r['validation_loss']) for r in rows],label='Validation')
  ax.set_title(name);ax.set_xlabel('Epoch');ax.set_ylabel('Loss');ax.set_xlim(1,100);ax.legend(frameon=False)
 save(f,a.output,'training_curves')
 f,axes=plt.subplots(4,5,figsize=(16,12),constrained_layout=True)
 for i,t in enumerate(['T1','T2','T3','T4']):
  rgb=np.array(Image.open(ROOT/'03_temporal_V66/assets/RGB'/f'{t}_RGB.png'))
  arrays=[rgb,colors(np.array(Image.open(ROOT/'03_temporal_V66/assets/evaluation_reference'/f'{t}_classes.png')))]+[colors(np.array(Image.open(a.v66/m/'full'/f'{t}_classes.png'))) for m in MODELS]
  for j,(ax,im) in enumerate(zip(axes[i],arrays)):
   ax.imshow(im,interpolation='antialiased' if j==0 else 'nearest');ax.set_xticks([]);ax.set_yticks([])
   if i==0:ax.set_title(['RGB','Reference','U-Net','DeepLabV3+','SegFormer-B0'][j])
   if j==0:ax.set_ylabel(t)
 save(f,a.output,'four_period_comparison')
 f,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
 for ax,(m,name) in zip(axes,MODELS.items()):
  _,cm=measure(a.v66/m/'full');cm=np.array(cm);norm=cm/cm.sum(1)[:,None];ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
  for i in range(3):
   for j in range(4):ax.text(j,i,f'{cm[i,j]}\n{norm[i,j]*100:.1f}%',ha='center',va='center',color='white' if norm[i,j]>.5 else 'black',fontsize=8)
  ax.set_xticks(range(4),['Persistent','New','Extension','Missed'],rotation=25);ax.set_yticks(range(3),['Persistent','New','Extension']);ax.set_title(name)
 save(f,a.output,'confusion_matrices')
 if a.v90:
  roi=[1530,590,1830,850];x0,y0,x1,y1=roi;f,axes=plt.subplots(4,4,figsize=(12,12),constrained_layout=True)
  for i,t in enumerate(['T1','T2','T3','T4']):
   rgb=np.array(Image.open(ROOT/'03_temporal_V66/assets/RGB'/f'{t}_RGB.png'))
   arrays=[rgb]+[colors(np.array(Image.open(folder/f'{t}_classes.png'))) for folder in [ROOT/'03_temporal_V66/assets/evaluation_reference',a.v66/'unet/full',a.v90]]
   for j,(ax,im) in enumerate(zip(axes[i],arrays)):
    ax.imshow(im[y0:y1,x0:x1],interpolation='antialiased' if j==0 else 'nearest');ax.set_xticks([]);ax.set_yticks([])
    if i==0:ax.set_title(['RGB','Reference','V66 U-Net','V90'][j])
    if j==0:ax.set_ylabel(t)
  save(f,a.output,'four_period_local_V66_V90')
if __name__=='__main__':main()
