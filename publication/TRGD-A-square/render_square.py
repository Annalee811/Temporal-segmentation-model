"""Square single-panel version; retains the numbers of the displayed merged A."""
from pathlib import Path
import json,csv,hashlib,shutil,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
HERE=Path(__file__).resolve().parent;ROOT=Path('/workspace/Temporal-segmentation-model');PUB=Path('/workspace/shared/downloads');K='TRGD_A_square'
source=ROOT/'work/TRGD_merged_A/attached_values/provenance.json';proof=json.loads(source.read_text());rows=proof['rows'];ks=['Precision','Recall','Dice','IoU'];colors={'new_root':'#D89500','existing_root':'#2D73A6'}
for p in Path('/tmp/v66-fonts').glob('*.TTF'):font_manager.fontManager.addfont(str(p))
plt.rcParams.update({'font.family':'Times New Roman','font.size':15,'axes.labelsize':17,'axes.labelweight':'bold','xtick.labelsize':16,'ytick.labelsize':15,'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black','pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(7.2,7.2),facecolor='white');fig.subplots_adjust(left=.15,right=.92,bottom=.20,top=.73);x=np.arange(4);data=[]
for cls in ['new_root','existing_root']:
 for i,row in enumerate(rows):
  vals=np.array([float(row[cls+'_'+k])*100 for k in ks]);ax.plot(x,vals,ls=(0,(4,2)) if i==0 else '-',marker='o' if i==0 else 's',ms=7.5,lw=1.8,color=colors[cls],markerfacecolor='white' if i==0 else colors[cls],markeredgewidth=1.5)
  dx,dy,ha={('existing_root',0):(-12,17,'right'),('existing_root',1):(12,3,'left'),('new_root',0):(-12,-19,'right'),('new_root',1):(12,-12,'left')}[cls,i]
  for xx,y,k in zip(x,vals,ks):
   ax.annotate(f'{y:.2f}',(xx,y),xytext=(dx,dy),textcoords='offset points',ha=ha,va='center',fontsize=12,fontweight='bold' if k=='Dice' else 'normal',color='black')
   data.append(dict(root_class=cls,model='V66 U-Net' if i==0 else 'TRGD (V90)',metric=k,score_percent=float(y)))
ax.set(xlim=(-.43,3.55),ylim=(0,113),ylabel='Score (%)');ax.set_xticks(x,ks);ax.set_yticks([0,25,50,75,100]);ax.tick_params(axis='x',length=0,pad=10);ax.grid(axis='y',alpha=.13,lw=.7);ax.set_axisbelow(True)
fig.text(.53,.977,'New and existing root classification',ha='center',va='top',fontsize=18,fontweight='bold')
classhandles=[Line2D([],[],color=colors[c],lw=2,label=n) for c,n in [('new_root','New root'),('existing_root','Existing root')]]
modelhandles=[Line2D([],[],color='#51565C',lw=1.8,ls=(0,(4,2)),marker='o',markerfacecolor='white',markeredgewidth=1.5,ms=7,label='V66 U-Net'),Line2D([],[],color='#51565C',lw=1.8,marker='s',ms=7,label='TRGD (V90)')]
fig.legend(handles=classhandles,loc='upper center',bbox_to_anchor=(.54,.92),ncol=2,frameon=False,fontsize=14,columnspacing=2.3,handlelength=2.5)
fig.legend(handles=modelhandles,loc='upper center',bbox_to_anchor=(.54,.855),ncol=2,frameon=False,fontsize=14,columnspacing=2.3,handlelength=2.5)
macro=proof['macro_Dice_percent'];fig.text(.535,.075,f'Macro Dice  {macro[0]:.2f}% → {macro[1]:.2f}%',ha='center',fontsize=17,fontweight='bold')
for ext,dpi in [('pdf',350),('svg',350),('png',450)]:fig.savefig(PUB/(K+'.'+ext),dpi=dpi,facecolor='white')
fig.savefig(PUB/(K+'.tiff'),dpi=600,facecolor='white',pil_kwargs={'compression':'tiff_lzw'});fig.savefig(PUB/(K+'_preview.png'),dpi=170,facecolor='white');plt.close(fig)
with (PUB/(K+'_plot_values.csv')).open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
proof.update(canvas_inches=[7.2,7.2],aspect_ratio='1:1',displayed_values_unchanged=True,source_figure='TRGD_A_merged_attached_values',legend='Two compact rows: root-class colours and model line/marker styles');(PUB/(K+'_provenance.json')).write_text(json.dumps(proof,indent=2)+'\n')
readme='''# 合并A图：正方形单图\n\n画布7.2×7.2英寸，宽高比1:1。PDF/SVG为矢量图，PNG为3240×3240像素，TIFF为4320×4320像素、600dpi、LZW无损压缩。\n\n仅调整排版，保持上一张显示图的数值：Macro Dice90.95%→93.59%。这组数值来自所附图对应的归档评估，不替换为另一套新版前景消融结果。模型名称统一为TRGD（V90）。\n\n颜色区分新生根/既有根；虚线空心圆表示V66 U-Net，实线实心方块表示TRGD（V90）。全部文字为黑色Times New Roman。\n\n- [PDF](TRGD_A_square.pdf)\n- [PNG](TRGD_A_square.png)\n- [TIFF，600dpi](TRGD_A_square.tiff)\n- [SVG](TRGD_A_square.svg)\n''';(PUB/(K+'_README.md')).write_text(readme)
with zipfile.ZipFile(PUB/(K+'_bundle.zip'),'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f in sorted(PUB.glob(K+'*')):
  if not f.name.endswith(('_bundle.zip','_checksums.json')):z.write(f,f.name)
 z.write(HERE/'render_square.py','code/render_square.py')
checks={f.name:dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in PUB.glob(K+'*') if not f.name.endswith('_checksums.json')};(PUB/(K+'_checksums.json')).write_text(json.dumps(checks,indent=2)+'\n');print('SQUARE_READY',len(checks),flush=True)
