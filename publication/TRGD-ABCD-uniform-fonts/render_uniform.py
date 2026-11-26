"""Rebuild user-composed A–D layout with one D-referenced 14pt text size.
Preserve legacy A/D and latest-foreground B/C as distinct documented comparisons.
"""
import csv,json,hashlib,sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
ROOT=Path('/workspace/Temporal-segmentation-model');HERE=Path(__file__).resolve().parent;PUB=Path('/workspace/shared/downloads');K='TRGD_ABCD_uniform_fonts';sys.path.insert(0,str(ROOT));from evaluate import measure_merged
for p in Path('/tmp/v66-fonts').glob('*.TTF'):font_manager.fontManager.addfont(str(p))
F=14;PANEL=20;plt.rcParams.update({'font.family':'Times New Roman','font.size':F,'axes.labelsize':F,'axes.titlesize':F,'axes.titleweight':'bold','axes.labelweight':'bold','xtick.labelsize':F,'ytick.labelsize':F,'legend.fontsize':F,'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black','pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.75})
fig=plt.figure(figsize=(14,14.2),facecolor='white');C={'new':'#E69F00','existing':'#3776B9'};OFF='#8A6B75';ON='#256B98'
def ft(x,y,t,bold=False,ha='left',size=F):return fig.text(x,y,t,fontsize=size,fontweight='bold' if bold else 'normal',ha=ha,va='center',color='black')
def ac(rect):return fig.add_axes(rect)
legacy=json.loads((ROOT/'work/TRGD_merged_A/attached_values/provenance.json').read_text());a0,a1=legacy['rows'];latest=json.loads((ROOT/'work/TRGD_latest_whole_ablation/evaluation.json').read_text());b0,b1=latest['metrics']
# A -- four series on the original percentage scale, preserving the supplied values.
axA=ac([.085,.760,.315,.195]);x=np.arange(4);keys=['Precision','Recall','Dice','IoU']
for label,col in [('new_root',C['new']),('existing_root',C['existing'])]:
 for i,row in enumerate([a0,a1]):
  v=np.array([float(row[label+'_'+k])*100 for k in keys]);axA.plot(x,v,color=col,lw=1.6,ls=(0,(4,2)) if i==0 else '-',marker='o' if i==0 else 's',ms=5.2,mfc='white' if i==0 else col,mew=1.3)
  dx,dy,ha={('existing_root',0):(-17,15,'center'),('existing_root',1):(17,2,'center'),('new_root',0):(-17,-17,'center'),('new_root',1):(17,-12,'center')}[label,i]
  for xx,yy,k in zip(x,v,keys):axA.annotate(f'{yy:.2f}',(xx,yy),xytext=(dx,dy),textcoords='offset points',ha=ha,va='center',fontsize=F,color='black',fontweight='bold' if k=='Dice' else 'normal')
axA.set(xlim=(-.48,3.58),ylim=(0,118),ylabel='Score (%)');axA.set_xticks(x,keys);axA.set_yticks([0,25,50,75,100]);axA.tick_params(axis='x',length=0,pad=8);axA.grid(axis='y',alpha=.12);axA.set_axisbelow(True)
cl=[Line2D([],[],color=C[k],lw=2,label=t) for k,t in [('new','New root'),('existing','Existing root')]];ml=[Line2D([],[],color='#51565C',ls='--',marker='o',mfc='white',ms=5,label='V66 U-Net'),Line2D([],[],color='#51565C',marker='s',ms=5,label='TRGD (V90)')]
l1=axA.legend(handles=cl,loc='lower center',bbox_to_anchor=(.52,.25),ncol=2,frameon=False,columnspacing=1,handlelength=2,handletextpad=.6);axA.add_artist(l1);axA.legend(handles=ml,loc='lower center',bbox_to_anchor=(.52,.08),ncol=2,frameon=False,columnspacing=1,handlelength=2,handletextpad=.6)
ft(.245,.979,'Two-class comparison',True,'center');ft(.245,.705,f"Macro Dice {float(a0['macro_Dice'])*100:.2f}% → {float(a1['macro_Dice'])*100:.2f}%",True,'center')
# B -- class-wise paired whole-process inference ablation, no component split.
axB=ac([.480,.760,.190,.195]);ys=[2,1,0];bk=['new_root_Dice','existing_root_Dice','macro_Dice']
for y,k in zip(ys,bk):
 a,b=b0[k]*100,b1[k]*100;axB.plot([a,b],[y,y],color='#9A9A9A',lw=1.5);axB.scatter(a,y,s=52,facecolors='white',edgecolors=OFF,lw=1.5,zorder=3);axB.scatter(b,y,s=48,marker='s',color=ON,zorder=3);axB.text(a,y+.18,f'{a:.2f}',ha='center',fontsize=F);axB.text(b,y-.27,f'{b:.2f}',ha='center',fontsize=F,fontweight='bold');axB.text(104.8,y,f'{b-a:+.2f}',ha='center',va='center',fontsize=F)
axB.set(xlim=(78,110),ylim=(-.5,2.6),yticks=ys,yticklabels=['New root','Existing root','Macro Dice'],xlabel='Dice (%)');axB.set_xticks([80,85,90,95,100]);axB.tick_params(axis='y',length=0,pad=9);axB.spines['left'].set_visible(False);axB.grid(axis='x',alpha=.12);axB.set_axisbelow(True);axB.text(104.8,2.45,'Δ (pp)',ha='center',fontsize=F);ft(.575,.979,'Class-wise Dice',True,'center')
# C -- new-root changes, including the decrease in recall.
axC=ac([.785,.760,.200,.195])
for y,k in zip([3,2,1,0],keys):
 a,b=b0['new_root_'+k]*100,b1['new_root_'+k]*100;axC.plot([a,b],[y,y],color='#9A9A9A',lw=1.5);axC.scatter(a,y,s=52,facecolors='white',edgecolors=OFF,lw=1.5,zorder=3);axC.scatter(b,y,s=48,marker='s',color=ON,zorder=3);axC.text(a,y+.17,f'{a:.2f}',ha='center',fontsize=F);axC.text(b,y-.29,f'{b:.2f}',ha='center',fontsize=F,fontweight='bold');axC.text(106.8,y,f'{b-a:+.2f}',ha='center',va='center',fontsize=F)
axC.set(xlim=(66,112),ylim=(-.5,3.6),yticks=[3,2,1,0],yticklabels=keys,xlabel='Score (%)');axC.set_xticks([70,80,90,100]);axC.tick_params(axis='y',length=0,pad=9);axC.spines['left'].set_visible(False);axC.grid(axis='x',alpha=.12);axC.set_axisbelow(True);axC.text(106.8,3.45,'Δ (pp)',ha='center',fontsize=F);ft(.885,.979,'New-root performance',True,'center')
fig.legend(handles=[Line2D([],[],marker='o',mfc='white',mec=OFF,mew=1.5,ms=6,ls='',label='TRGD off'),Line2D([],[],marker='s',color=ON,ms=6,ls='',label='TRGD on')],loc='center',bbox_to_anchor=(.73,.705),ncol=2,frameon=False,columnspacing=3)
for t,xx in [('A',.025),('B',.435),('C',.715)]:ft(xx,.973,t,True,size=PANEL)
# D -- same original420×320pixel crops; only font sizes and the V90 name change.
assets=ROOT/'03_temporal_V66/assets';ref=ROOT/'work/V66_two_class_annotations/evaluation_reference';base=ROOT/'work/V66_merged/unet/full';full=ROOT/'work/V90_optimized/prediction/full';roi=(865,650,1285,970);left=.13;right=.965;gapx=.006;gapy=.0045;tilew=(right-left-3*gapx)/4;tileh=tilew*14/14.2*320/420;bottom=.035;top=bottom+4*tileh+3*gapy
ft(.082,top+.024,'D',True,size=PANEL)
for j,title in enumerate(['Original RGB','Human reference','V66 U-Net','TRGD (V90)']):ft(left+j*(tilew+gapx)+tilew/2,top+.024,title,True,'center')
load=lambda p:np.array(Image.open(p).crop(roi));image_audit=[]
for i,t in enumerate(['T1','T2','T3','T4']):
 rgb=load(assets/'RGB'/(t+'_RGB.png'));tiles=[rgb];labels_list=[]
 for directory in [ref,base,full]:
  labels=load(directory/(t+'_classes.png'));assert np.isin(labels,[0,4,255] if i==0 else [0,1,2,255]).all();ov=rgb.copy()
  for code,col in {1:[55,118,185],2:[230,159,0],4:[255,255,255]}.items():ov[labels==code]=col
  tiles.append(ov);labels_list.append(labels)
 y=top-(i+1)*tileh-i*gapy;ft(.09,y+tileh/2,t,True,'center')
 for j,tile in enumerate(tiles):
  ax=ac([left+j*(tilew+gapx),y,tilew,tileh]);ax.imshow(tile,interpolation='none');ax.set_axis_off()
  if i==j==0:
   ax.plot([18,118],[302,302],color='white',lw=1.5,solid_capstyle='butt');ax.text(68,281,'100 px',va='bottom',ha='center',fontsize=F,color='black',bbox=dict(facecolor='white',edgecolor='none',pad=.5))
 image_audit.append(dict(period=t,ROI=list(roi),native_crop_width=420,native_crop_height=320,RGB_pixel_SHA256=hashlib.sha256(rgb.tobytes()).hexdigest(),V66_TRGD_foreground_difference_pixels=int(((labels_list[1]>0)&(labels_list[1]!=255)!=(labels_list[2]>0)&(labels_list[2]!=255)).sum())))
for ext,dpi in [('pdf',350),('svg',350),('png',300)]:fig.savefig(PUB/(K+'.'+ext),dpi=dpi,facecolor='white');print('SAVED',ext,flush=True)
fig.savefig(PUB/(K+'.tiff'),dpi=600,facecolor='white',pil_kwargs={'compression':'tiff_adobe_deflate'});fig.savefig(PUB/(K+'_preview.png'),dpi=115,facecolor='white');plt.close(fig)
proof=dict(body_font='Times New Roman',body_font_pt=F,panel_letter_pt=PANEL,font_colour='black',D_column_header_pt=F,A=dict(source='archived V66 U-Net and V90 comparison',macro_Dice=[float(a0['macro_Dice']),float(a1['macro_Dice'])],metrics=[a0,a1]),BC=dict(source='latest V35 foreground whole-process frozen inference ablation',metrics=[b0,b1],reference_root_pixels=468203,missed_root_pixels=17438),D=dict(source_directories=[str(p.relative_to(ROOT)) for p in [ref,base,full]],ROI=list(roi),native_pixels_preserved=True,images=image_audit),evaluation_versions_not_interchangeable=True,metric_domain='Previously annotated reference-root pixels of development sequence0353-2-B3, T2–T4; background excluded; no independent temporal test set')
(PUB/(K+'_provenance.json')).write_text(json.dumps(proof,indent=2)+'\n');print('UNIFORM_ABCD_READY',flush=True)
