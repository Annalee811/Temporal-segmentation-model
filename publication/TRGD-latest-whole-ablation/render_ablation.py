"""SCI whole-process ablation figure; frozen latest V35 support in both controls."""
import json,hashlib,shutil,csv,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import fitz
HERE=Path(__file__).resolve().parent;PUB=Path('/workspace/shared/downloads');PREFIX='TRGD_latest_whole_pipeline_ablation';R=json.loads((HERE/'evaluation.json').read_text());off,on=R['metrics'];O='#8A6B75';N='#256B98'
for p in Path('/tmp/v66-fonts').glob('*.TTF'):font_manager.fontManager.addfont(str(p))
plt.rcParams.update({'font.family':'Times New Roman','font.size':14,'axes.labelsize':16,'axes.labelweight':'bold','axes.titlesize':20,'axes.titleweight':'bold','xtick.labelsize':13,'ytick.labelsize':14,'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black','pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(15.8,5.8),facecolor='white');gs=fig.add_gridspec(1,3,width_ratios=[1.25,1,1.2],left=.035,right=.985,top=.76,bottom=.24,wspace=.40);axes=[fig.add_subplot(gs[i]) for i in range(3)];ax=axes[0];ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(-.035,1.02)
def box(x,y,w,h,text,color='#F4F5F6',fs=13):
 p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=.02',lw=1,edgecolor='#535353',facecolor=color);ax.add_patch(p);ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,color='black')
def arrow(start,end):ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',lw=1.25,color='#333333'))
box(.02,.855,.96,.13,'Matched RGB + fixed V35 foreground',fs=13);arrow((.5,.85),(.5,.77));box(.13,.65,.74,.115,'Frozen V66 temporal priors',fs=13)
arrow((.34,.64),(.23,.56));arrow((.67,.64),(.76,.56));box(.02,.29,.40,.25,'TRGD off\nBypass full process',color='#F4ECEF',fs=14);box(.53,.20,.45,.35,'TRGD on (full)\nConvLSTM + GAT\n↓ Axis → Junction',color='#EAF2F7',fs=13.5)
arrow((.22,.28),(.34,.13));arrow((.75,.19),(.66,.13));box(.03,.0,.94,.115,'New root / Existing root',fs=14);ax.set_title('Paired workflow',pad=19)
# Paired differences are descriptive; do not invent replicate errors or tests.
ax=axes[1];ys=np.array([2,1,0]);keys=['new_root_Dice','existing_root_Dice','macro_Dice'];labels=['New root','Existing root','Macro Dice']
for y,k in zip(ys,keys):
 a,b=off[k]*100,on[k]*100;ax.plot([a,b],[y,y],color='#9A9A9A',lw=2,zorder=1);ax.scatter(a,y,s=85,facecolor='white',edgecolor=O,lw=1.8,zorder=3);ax.scatter(b,y,s=80,marker='s',color=N,zorder=3)
 ax.text(a,y+.18,f'{a:.2f}',ha='center',fontsize=12.5);ax.text(b,y-.26,f'{b:.2f}',ha='center',fontsize=12.5,fontweight='bold');ax.text(102.8,y,f'+{b-a:.2f}',va='center',ha='center',fontsize=12)
ax.set(yticks=ys,yticklabels=labels,xlim=(78,106),ylim=(-.5,2.6),xlabel='Dice (%)');ax.set_xticks([80,85,90,95,100]);ax.grid(axis='x',alpha=.17);ax.set_axisbelow(True);ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,pad=9);ax.set_title('Class-wise Dice',pad=19);ax.text(102.8,2.45,'Δ (pp)',ha='center',fontsize=12)
ax=axes[2];ys=np.arange(4)[::-1];keys=['Precision','Recall','Dice','IoU']
for y,k in zip(ys,keys):
 a,b=off['new_root_'+k]*100,on['new_root_'+k]*100;ax.plot([a,b],[y,y],color='#9A9A9A',lw=2,zorder=1);ax.scatter(a,y,s=85,facecolor='white',edgecolor=O,lw=1.8,zorder=3);ax.scatter(b,y,s=80,marker='s',color=N,zorder=3)
 ax.text(a,y+.16,f'{a:.2f}',ha='center',fontsize=12);ax.text(b,y-.28,f'{b:.2f}',ha='center',fontsize=12,fontweight='bold');ax.text(105.5,y,f'{b-a:+.2f}',va='center',ha='center',fontsize=12)
ax.set(yticks=ys,yticklabels=keys,xlim=(66,109),ylim=(-.5,3.6),xlabel='Score (%)');ax.set_xticks([70,80,90,100]);ax.grid(axis='x',alpha=.17);ax.set_axisbelow(True);ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,pad=9);ax.set_title('New-root performance',pad=19);ax.text(105.5,3.45,'Δ (pp)',ha='center',fontsize=12)
fig.text(.5,.985,'TRGD whole-pipeline ablation',ha='center',va='top',fontsize=21,fontweight='bold');fig.text(.5,.885,'Latest V35 foreground (epoch 91) · One annotated sequence, T2–T4 · 468,203 reference-root pixels',ha='center',fontsize=13)
legend=[Line2D([],[],marker='o',markerfacecolor='white',markeredgecolor=O,markeredgewidth=1.8,ls='',markersize=8,label='TRGD off (V66 prior only)'),Line2D([],[],marker='s',color=N,ls='',markersize=8,label='TRGD on (full process)')];fig.legend(handles=legend,ncol=2,loc='center',bbox_to_anchor=(.5,.115),frameon=False,fontsize=14,columnspacing=3)
fig.text(.5,.035,'Both conditions retain the same 17,438 missed reference-root pixels; T1 provides context and is not scored.',ha='center',fontsize=12)
letters=[]
for i,ax in enumerate(axes):
 p=ax.get_position();letters.append(fig.text(max(.006,p.x0-.035),.79,'ABC'[i],fontsize=27,fontweight='bold',va='bottom'))
for ext,dpi in [('pdf',350),('svg',350),('png',450)]:fig.savefig(PUB/(PREFIX+'.'+ext),dpi=dpi,facecolor='white')
fig.savefig(PUB/(PREFIX+'.tiff'),dpi=600,facecolor='white',pil_kwargs={'compression':'tiff_lzw'});fig.savefig(HERE/'preview.png',dpi=150,facecolor='white')
for label,text in zip(letters,['E','F','G']):label.set_text(text)
extra=HERE/'ablation_EFG.pdf';fig.savefig(extra,dpi=350,facecolor='white');plt.close(fig)
# Preserve A–D geometry and vector text, append the whole-process ablation as E–G.
main=fitz.open(PUB/'V35_updated_models_ABCD.pdf');ab=fitz.open(extra);combo=fitz.open();w=main[0].rect.width;h=main[0].rect.height;bh=ab[0].rect.height*w/ab[0].rect.width;page=combo.new_page(width=w,height=h+bh);page.show_pdf_page(fitz.Rect(0,0,w,h),main,0);page.show_pdf_page(fitz.Rect(0,h,w,h+bh),ab,0);combo.save(PUB/'V35_TRGD_with_whole_ablation.pdf',garbage=4,deflate=True);page.get_pixmap(matrix=fitz.Matrix(300/72,300/72),alpha=False).save(PUB/'V35_TRGD_with_whole_ablation.png');page.get_pixmap(matrix=fitz.Matrix(100/72,100/72),alpha=False).save(PUB/'V35_TRGD_with_whole_ablation_preview.png')
# Alternative two-page version keeps both figures at a comfortable reading size.
pages=fitz.open();pages.insert_pdf(main);pages.insert_pdf(fitz.open(PUB/(PREFIX+'.pdf')));pages.save(PUB/'V35_TRGD_main_and_ablation_2pages.pdf',garbage=4,deflate=True)
for name in ['metrics.csv','per_period_metrics.csv','conditional_metrics.csv','paired_prediction_audit.csv','evaluation.json']:shutil.copy2(HERE/name,PUB/(PREFIX+'_'+name))
caption='''Whole-pipeline inference ablation of deployed Temporal Root Graph Decoding (TRGD, V90) with the latest V35 root support. TRGD off uses frozen V66 temporal priors and bypasses the entire V90 process. TRGD on applies the coupled ConvLSTM+GAT, Axis root-axis constraint and Junction connection constraint as one process. All components are switched as a bundle; no component-specific ablation or new training is performed. (A/E) Matched control design. (B/F) Dice of new roots, existing roots and their unweighted mean (Macro Dice). (C/G) New-root precision, recall, Dice and IoU. Δ is a percentage-point difference; a negative recall change is shown explicitly.\n\nBoth conditions use identical matched RGB pixels and the V35 epoch91 foreground, without registration, image rescaling or foreground changes. Foreground identity is checked across all66 available images at18observation positions. Temporal accuracy is scored only on the available previously annotated sequence0353-2-B3 atT2–T4, using468203reference-root pixels. T1 is context only. The same17438reference-root pixels missed by V35 are retained as missed predictions in both controls. Existing and legacy extension labels are merged; only new-root and existing-root classes are scored. Background pixels are outside this temporal metric domain, so these precision/recall values describe classification on the reference-root support, not full root/background segmentation. A separate conditional analysis on450765detected reference-root pixels is supplied and is not substituted for the main metric.\n\nThis is a development-sequence inference ablation of the frozen deployed transfer adapter. Its reconstructed graph preparation differs from the archived original preprocessing, as listed in evaluation.json. The original V90 weights are preserved, and there is no independent temporal test set or repeated training experiment. These temporal metrics must not be confused with V35 binary root/background Dice89.85% computed on48monitoring parent images. The combined A–G figure retains these distinct evaluation domains. No confidence interval, significance claim or inferred AUC is added from hard class maps.\n'''
import re
identifiers=['V35','V66','V90','T1','T2','T3','T4','0353-2-B3']
for i,token in enumerate(identifiers):caption=caption.replace(token,chr(0xE000+i))
caption=re.sub(r'(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])',' ',caption)
for i,token in enumerate(identifiers):caption=caption.replace(chr(0xE000+i),token)
(PUB/(PREFIX+'_caption.txt')).write_text(caption)
cn='''TRGD（V90）整体流程消融：使用最新V35第91轮根系前景、冻结V66时序先验和冻结V90权重。TRGD off绕过整套流程；TRGD on将ConvLSTM+GAT→Axis根轴约束→Junction交叉连接约束作为同一个整体开启。不拆分各组件，不重新训练。\n\n18个观测位置、66张现有图像均完成前景一致性核验。准确度仅使用0353-2-B3已有时序标记的T2—T4，合计468203个参考根系像素；T1仅提供上下文。两组同样保留17438个V35漏检的参考根像素，不将漏检剔除后冒充主指标。背景不属于本次时序分类计分域。仅在450765个检出参考根像素上的条件指标另行保存。\n\nTRGD off→on：新生根Dice83.01%→88.07%；既有根Dice95.28%→96.55%；Macro Dice89.14%→92.31%，提升3.17个百分点。新生根Precision82.01%→93.18%，Recall84.02%→83.50%。\n\n本次为既有开发序列上、当前迁移部署接口的成套推理消融，使用已完成的配对预测结果重算；不代表重新训练的组件消融，也不是独立时序测试。接口与归档原前处理的实际差异见evaluation.json。V35根系/背景Dice仍为89.85%，与此处两类根时序Macro Dice分别报告。\n'''
(PUB/(PREFIX+'_caption_CN.txt')).write_text(cn)
readme='''# TRGD整体流程消融：最新V35前景\n\n- [整体消融PDF](TRGD_latest_whole_pipeline_ablation.pdf)\n- [整体消融PNG](TRGD_latest_whole_pipeline_ablation.png)\n- [整体消融TIFF，600dpi](TRGD_latest_whole_pipeline_ablation.tiff)\n- [整体消融SVG](TRGD_latest_whole_pipeline_ablation.svg)\n- [主图加消融，A–G合并PDF](V35_TRGD_with_whole_ablation.pdf)\n- [主图加消融，A–G合并PNG](V35_TRGD_with_whole_ablation.png)\n- [主图和消融双页PDF](V35_TRGD_main_and_ablation_2pages.pdf)\n\n'''+cn+'\n标题和坐标使用黑色Times New Roman，PDF嵌入字体，三组件作为整体进行推理开关对照。所有模型权重、原始标记和既有结果保留。\n'
(PUB/(PREFIX+'_README.md')).write_text(readme)
# Compact native-resolution maps for exact reproduction; no extra RGB copies.
with zipfile.ZipFile(PUB/(PREFIX+'_paired_class_maps.zip'),'w',zipfile.ZIP_DEFLATED) as z:
 for r in list(csv.DictReader(open(HERE/'paired_prediction_audit.csv',encoding='utf-8-sig'))):
  pos,t=r['position'],r['period'];z.write(HERE/'trgd_off'/pos/(t+'_classes.png'),f'TRGD_off/{pos}/{t}_classes.png');z.write(Path(r['TRGD_on_source']),f'TRGD_on/{pos}/{t}_classes.png')
 z.write(HERE/'paired_prediction_audit.csv','paired_prediction_audit.csv')
files=list(PUB.glob(PREFIX+'*'))+list(PUB.glob('V35_TRGD_*ablation*'))
with zipfile.ZipFile(PUB/(PREFIX+'_bundle.zip'),'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(set(files)):
  if not p.name.endswith(('_bundle.zip','_checksums.json')):z.write(p,p.name)
 for f in ['evaluate_whole_pipeline.py','render_ablation.py']:z.write(HERE/f,'code/'+f)
files=list(PUB.glob(PREFIX+'*'))+list(PUB.glob('V35_TRGD_*ablation*'));checks={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in set(files) if not p.name.endswith('_checksums.json')};(PUB/(PREFIX+'_checksums.json')).write_text(json.dumps(checks,indent=2)+'\n');print('ABLATION_FIGURES_READY',len(checks),flush=True)
