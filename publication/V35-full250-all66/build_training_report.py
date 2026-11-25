import csv,hashlib,json,shutil,sys,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path('/workspace/Temporal-segmentation-model');OUT=ROOT/'work/V35_full250_all66';SOURCE=ROOT/'work/V35_310_100epochs';PUB=Path('/workspace/shared/downloads');PREFIX='V35_full250_all66_'
sys.path.insert(0,str(ROOT/'02_root_background'))
from models import RootUNet
from infer_full_250 import predict_image
from train_merged_310 import metrics
import common as c
DONE=c.read(OUT/'completion.json');assert DONE['epochs']==100 and not DONE['early_stopping']
H=c.csvread(OUT/'history.csv');assert len(H)==101 and [int(r['epoch']) for r in H]==list(range(101))
assert all(int(r['patches_visited'])==1048 and int(r['new_patches_visited'])==264 for r in H[1:])
PARENTS=c.csvread(OUT/'data/parents.csv');AUDIT=c.read(OUT/'data_audit.json');SELECTION=c.read(OUT/'selection.json')
assert AUDIT['all_parent_pixels_covered'] and AUDIT['annotated_root_pixels_discarded']==0
for name,digest in DONE['preserved_weights_SHA256'].items():assert c.sha(SOURCE/'weights'/name)==c.sha(OUT/'weights'/name)==digest
for name in ['Times.TTF','Timesbd.TTF','Timesi.TTF']:font_manager.fontManager.addfont('/tmp/v66-fonts/'+name)
plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black','axes.edgecolor':'black','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
epoch=np.array([int(r['epoch']) for r in H]);get=lambda k:np.array([float(r[k]) if r[k] not in ['',None] else np.nan for r in H])
fig,axes=plt.subplots(1,3,figsize=(10.8,3.4));colors=['#356F9F','#D08A22'];best=DONE['selected_epoch']
axes[0].plot(epoch,get('train_loss'),color=colors[0],lw=1.6);axes[0].set_ylabel('Weighted training objective')
for source,color,name in [('old',colors[0],'Original training crops'),('new',colors[1],'Supplemental training crops')]:
 axes[1].plot(epoch,get(source+'_train_Dice')*100,color=color,lw=1.6,label=name)
axes[1].set_ylabel('Training Dice (%)');axes[1].legend(frameon=False,fontsize=8)
for key,color,name in [('Dice',colors[0],'Dice'),('Recall',colors[1],'Recall'),('Precision','#666666','Precision')]:
 axes[2].plot(epoch,get('validation_'+key)*100,color=color,lw=1.6,label=name)
axes[2].set_ylabel('Old-data internal validation (%)');axes[2].legend(frameon=False,fontsize=8)
for i,ax in enumerate(axes):
 ax.set_xlabel('Completed epoch');ax.set_xlim(0,100);ax.spines[['top','right']].set_visible(False)
 ax.axvline(best,color='0.55',ls=':',lw=.9);ax.text(-.18,1.05,chr(65+i),transform=ax.transAxes,fontweight='bold',fontsize=15)
 ax.grid(axis='y',color='.91',lw=.5)
fig.tight_layout(rect=(0,.03,1,.95));fig.suptitle(f'V35: complete 100-epoch training | 250 × 250 px | selected checkpoint: epoch {best}',fontsize=11)
for ext in ['pdf','svg','png']:fig.savefig(PUB/(PREFIX+'training.'+ext),dpi=600,bbox_inches='tight')
plt.close(fig)
caption='V35 binary root/background segmentation after complete native250px coverage of310 source images (244 original,66 supplemental). Four250px corners preserve each256px original and four disjoint250px quadrants cover each500px supplemental image.1240 correlated crops are not1240 independent samples. All264 supplemental quadrants are forced into training with spatially overlapping old-data groups. There are1048 training crops (784old,264supplemental) and192old-only internal-validation crops. Every training crop is visited in each of100 completed epochs, without early stopping or additional spatial cropping. Supplemental samples receive sixfold loss weight and an additional recall-oriented Tversky term. (A) Weighted training objective. (B) Source-specific training Dice on augmented crops. (C) Old-only internal-validation Dice,recall,precision on reassembled parent pixels. No supplemental image is treated as validation. Historical warm-start exposure prevents independent-test interpretation. The dotted line marks selected checkpoint,not training termination.'

(OUT/'figure_caption.txt').write_text(caption)
# Actual full500px predictions and paired diagnostics on every submitted source image.
torch.set_num_threads(4);net=RootUNet().to(memory_format=torch.channels_last);records=[];preds={};diag=OUT/'supplemental_predictions';diag.mkdir(exist_ok=True)
new=[r for r in PARENTS if r['source']=='submitted_66'];assert len(new)==66 and all(r['split']=='train' for r in new)
for label,weights in [('initial',OUT/'weights/V35_310_best.pt'),('selected',OUT/'weights/V35_full250_best.pt')]:
 net.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True)['state_dict'])
 for r in new:
  name=r['image_id'];rgb=np.array(Image.open(SOURCE/'data/RGB'/name).convert('RGB'));gt=np.array(Image.open(SOURCE/'data/mask'/name));p=predict_image(net,rgb);pred=p>=.5;cc=c.counts(gt,pred);records.append(dict(model=label,image_id=name,source=r['source'],position=r['position'],period=r['period'],split=r['split'],**metrics(cc)))
  preds[label,name]=pred
  if label=='selected':
   Image.fromarray(pred.astype('uint8')*255).save(diag/(Path(name).stem+'_root_mask.png'))
   overlay=rgb.copy();overlay[pred]=(rgb[pred]*.25+np.array([240,195,0])*.75).astype('uint8');Image.fromarray(overlay).save(diag/(Path(name).stem+'_root_overlay.png'))
   Image.fromarray(gt).save(diag/(Path(name).stem+'_submitted_mask.png'))
 print('SUPPLEMENTAL_DIAGNOSTICS',label,66,flush=True)
c.csvout(OUT/'supplemental_before_after_per_image.csv',records)
summary=[]
for label in ['initial','selected']:
 for split in ['train']:
  rr=[r for r in records if r['model']==label and r['split']==split];cc=np.array([sum(int(r[k]) for r in rr) for k in ['TP','FP','FN','TN']]);summary.append(dict(model=label,split=split,images=len(rr),**metrics(cc)))
c.csvout(OUT/'supplemental_before_after_summary.csv',summary)
examples=['0353-1-B3_ROI500_T1.png','Q1-1-B2_ROI500_T4.png','A017-3-B3_ROI500_T4.png','0353-2-B3_ROI500_T2.png']
fig,axes=plt.subplots(4,4,figsize=(10,10.7))
for i,name in enumerate(examples):
 rgb=np.array(Image.open(SOURCE/'data/RGB'/name).convert('RGB'));gt=np.array(Image.open(SOURCE/'data/mask'/name))==255
 for j,mask in enumerate([None,gt,preds['initial',name],preds['selected',name]]):
  im=rgb.copy()
  if mask is not None:im[mask]=[240,195,0]
  axes[i,j].imshow(im,interpolation='none');axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
  for spine in axes[i,j].spines.values():spine.set_visible(False)
  if i==0:axes[i,j].set_title(['Original RGB','Submitted roots','Previous V35','Updated V35'][j],fontsize=11)
 axes[i,0].set_ylabel(name.replace('_ROI500','').replace('.png',''),fontsize=9)
fig.tight_layout(pad=.5)
for ext in ['pdf','png']:fig.savefig(PUB/(PREFIX+'examples.'+ext),dpi=400,bbox_inches='tight')
plt.close(fig)
# All figures contain real model predictions. No annotation is pasted into a prediction.
final=c.csvread(OUT/'final_validation_metrics.csv');selected=next(r for r in final if r['model']=='selected');last=next(r for r in final if r['model']=='epoch100')
sub=c.csvread(OUT/'selected_validation_by_source.csv');assert len(sub)==1 and sub[0]['source']=='original_244';oldval=sub[0]
base=c.read(OUT/'baseline_validation.json')
readme=f'''# V35：补充66张、264个250px裁块全部用于训练

本轮实际完成100轮，无早停。从上一版V35_310_best.pt（前次100轮实验选定第32轮）初始化全部卷积/仿射权重，保留BatchNorm运行统计。原始第77轮及上一版第32轮、第100轮权重均完整保留，不以原第77轮作为本轮终点。

310张源图=旧244张256×256+补充66张500×500。500px完整四分为264个250px块；256px旧图使用偏移0和6的四角250px块共976个，保留边缘。总1240块，既不缩放，也不丢弃源像素。重叠子块不能视为独立样本。

**全部66张补充图像、264个子块均在训练集中，不留补充验证图。** 先强制补充图及与其空间相交的旧图组进入训练，再从剩余旧图的完整空间组中选48张父图验证。训练1048块（784旧+264补充），内部验证192块（全部旧数据），对应262张训练父图、48张旧验证父图。无独立测试。

每轮完整使用1048块一次，不再随机截小窗口。共104800次训练裁块访问，其中补充26400次。补充损失权重为旧样本6倍，占加权裁块暴露约66.89%；附加0.5Tversky项（假阴性权重0.7）。保留旧数据训练以维持既有识别能力。旋转、翻转、亮度和gamma增强不截去标记区域。

旧数据验证以父图原像素重建，250px滑窗、188px步长、Hann加权，原像素只计一次，阈值0.5。仅按旧数据内部验证Dice选权重，选中第{best}轮，另保留真实第100轮权重。补充图上的指标全部是训练拟合诊断，不用作验证或独立测试成绩。暖启动有历史来源接触，旧数据验证也不能解释为独立泛化。

## 真实结果

旧数据内部验证：上一版初始化Dice {base['overall']['Dice']*100:.2f}% → 本轮选定权重Dice {float(selected['Dice'])*100:.2f}%；选定权重Precision {float(selected['Precision'])*100:.2f}%，Recall {float(selected['Recall'])*100:.2f}%，IoU {float(selected['IoU'])*100:.2f}%。第100轮旧数据验证Dice {float(last['Dice'])*100:.2f}%。

66张训练来源的完整500px前后预测与逐图拟合指标在supplemental_before_after文件中。模型输出由网络计算，不用提交标记覆盖预测。所有时序根类别统一视为根系，标记按上传状态保留。

training.pdf/svg/png使用真实100轮记录，黑字Times New Roman。(A)加权训练目标；(B)旧/补充训练Dice；(C)仅旧数据内部验证指标。examples.pdf/png对比同一RGB、提交标记、上一版模型和本轮模型。supplemental_predictions.zip含66张500px叠加PNG及根系mask，原尺寸不变。

## 文件和推理

V35_full250_all66_best_model.zip提供选定权重、模型与训练/推理代码、协议。三份旧权重分别放在backup_original77、backup_previous_best32、backup_previous_epoch100三个ZIP中，避免单个文件过大。epoch100_model.zip提供真实第100轮权重。dataset_old_A.zip、dataset_old_B.zip、dataset_new66.zip共1240个250px RGB/mask；dataset_parents.zip提供310张原父图及mask，便于重建验证与复现。

```bash
python -m pip install -r requirements.txt
python infer_full_250.py --weights weights/V35_full250_best.pt --image your_image.png --output predictions
# 第100轮：--weights weights/V35_full250_epoch100.pt
```

推理输出背景0/根系255和概率，不自动覆盖已发布V90。复现时四个数据包解压得到run/与source/，把模型包三份旧权重复制至run/weights/，代码放同一目录，运行：

```bash
python train_full_250.py --source source --output run --batch 8 --threads 4 --seed 20261125
```

来源SHA、裁剪位置、分组、每张源图完整覆盖证明、逐epoch使用数量、训练协议和模型SHA随包提供。
'''

(OUT/'README.md').write_text(readme);(PUB/(PREFIX+'README.md')).write_text(readme)
req=OUT/'requirements.txt';req.write_text('torch==2.8.0\ntorchvision==0.23.0\ntransformers==4.57.6\nnumpy==2.2.6\nPillow==11.3.0\nmatplotlib\n')
code=[(ROOT/'02_root_background'/f,f) for f in ['models.py','common.py','train_merged_310.py','prepare_full_250.py','train_full_250.py','infer_full_250.py']]+[(req,'requirements.txt')]
common=[(OUT/f,f) for f in ['README.md','data_audit.json','protocol.json','selection.json','completion.json','history.csv','figure_caption.txt','final_validation_metrics.csv','supplemental_before_after_summary.csv','supplemental_before_after_per_image.csv']]
figs=[(PUB/(PREFIX+'training.'+ext),'figures/training.'+ext) for ext in ['pdf','svg','png']]
def archive(name,items):
 target=PUB/(PREFIX+name+'.zip')
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  seen=set()
  for path,arc in items:
   if arc in seen:continue
   seen.add(arc);z.write(path,arc)
 assert target.stat().st_size<100*1048576,(target,target.stat().st_size)
 with zipfile.ZipFile(target) as z:assert z.testzip() is None
 print('PACKAGED',target.name,target.stat().st_size,flush=True)
archive('best_model',common+code+figs+[(OUT/'weights/V35_full250_best.pt','weights/V35_full250_best.pt')])
for name,weight in [('backup_original77','V35_original_epoch77.pt'),('backup_previous_best32','V35_310_best.pt'),('backup_previous_epoch100','V35_310_epoch100.pt')]:
 archive(name,[(OUT/'weights'/weight,'weights/'+weight),(OUT/'protocol.json','protocol.json'),(OUT/'data_audit.json','data_audit.json')])
archive('epoch100_model',common+code+figs+[(OUT/'weights/V35_full250_epoch100.pt','weights/V35_full250_epoch100.pt')])
tiles=c.csvread(OUT/'data/manifest.csv');old=[r for r in tiles if r['source']=='original_244'];newtiles=[r for r in tiles if r['source']=='submitted_66']
metadata=[(OUT/'data'/f,'run/data/'+f) for f in ['manifest.csv','parents.csv','coverage.csv']]+[(OUT/f,'run/'+f) for f in ['data_audit.json','uploaded_annotation.json']]
for name,rs in [('dataset_old_A',old[:488]),('dataset_old_B',old[488:]),('dataset_new66',newtiles)]:
 archive(name,metadata+[(OUT/'data'/kind/r['image_id'],'run/data/'+kind+'/'+r['image_id']) for r in rs for kind in ['RGB','mask']])
archive('dataset_parents',[(SOURCE/'data'/kind/r['image_id'],'source/data/'+kind+'/'+r['image_id']) for r in PARENTS for kind in ['RGB','mask']]+[(SOURCE/'data/manifest.csv','source/data/manifest.csv')])
archive('supplemental_predictions',[(p,p.name) for p in diag.glob('*.png')]+[(OUT/f,f) for f in ['supplemental_before_after_per_image.csv','supplemental_before_after_summary.csv']])
archive('results',common+code+figs+[(PUB/(PREFIX+'examples.'+ext),'figures/examples.'+ext) for ext in ['pdf','png']]+[(p,'epoch_records/'+p.name) for p in (OUT/'epoch_records').glob('*.json')]+[(p,p.name) for p in OUT.glob('*validation*.csv')]+[(OUT/'training.log','training.log')])
for f in ['history.csv','data_audit.json','protocol.json','selection.json','completion.json','final_validation_metrics.csv','supplemental_before_after_summary.csv','supplemental_before_after_per_image.csv']:
 shutil.copy2(OUT/f,PUB/(PREFIX+f))
import fitz
with fitz.open(PUB/(PREFIX+'training.pdf')) as document:
 assert len(document)==1;assert any('Times' in font[3] for font in document.get_page_fonts(0))
 document[0].get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False).save(str(OUT/'training_preview.png'))
files=[p for p in PUB.glob(PREFIX+'*') if p.is_file() and p.name!=PREFIX+'checksums.json']
checks={p.name:dict(bytes=p.stat().st_size,sha256=c.sha(p)) for p in files}
(PUB/(PREFIX+'checksums.json')).write_text(json.dumps(checks,indent=2)+'\n')
(OUT/'exports_complete.json').write_text(json.dumps(dict(files=len(files),bytes=sum(p.stat().st_size for p in files),epochs=100),indent=2)+'\n')
print('FULL250_REPORT_AND_PACKAGES_COMPLETE',len(files),flush=True)
