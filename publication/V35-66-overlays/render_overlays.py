import csv,hashlib,json,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import fitz
ROOT=Path('/workspace/Temporal-segmentation-model');run=ROOT/'work/V35_310_100epochs';full=ROOT/'work/V35_310_all18_predictions';out=ROOT/'work/V35_66_overlay';pub=Path('/workspace/shared/downloads');out.mkdir(exist_ok=True)
for sub in ['submitted_labels','model_predictions','submitted_masks','prediction_masks']:(out/sub).mkdir(exist_ok=True)
rows=[r for r in csv.DictReader(open(run/'data/manifest.csv',encoding='utf-8-sig')) if r['source']=='submitted_66'];assert len(rows)==66
complete=json.load(open(full/'completion.json'));assert complete['images']==66;records=[];cache={};lookup={}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for r in rows:
 name=r['image_id'];pos=r['position'];t=r['period'];im=np.array(Image.open(run/'data/RGB'/name).convert('RGB'));g=np.array(Image.open(run/'data/mask'/name));x,y=int(r['x']),int(r['y'])
 key=(pos,t)
 if key not in cache:cache[key]=np.array(Image.open(full/pos/(t+'_root_mask.png')))
 pred=cache[key][y:y+500,x:x+500];assert im.shape==(500,500,3) and g.shape==pred.shape==(500,500)
 for array,folder in [(g,'submitted_labels'),(pred,'model_predictions')]:
  a=im.copy();m=array==255;a[m]=(im[m].astype('float32')*.25+np.array([255,193,7],np.float32)*.75).astype('uint8');Image.fromarray(a).save(out/folder/name);assert np.array_equal(a[~m],im[~m]);assert Image.open(out/folder/name).size==(500,500)
 Image.fromarray(g).save(out/'submitted_masks'/name);Image.fromarray(pred).save(out/'prediction_masks'/name)
 rec=dict(image_id=name,position=pos,period=t,material=r['material'],dataset_split=r['split'],width=500,height=500,x=x,y=y,submitted_root_pixels=int((g==255).sum()),predicted_root_pixels=int((pred==255).sum()),RGB_SHA256=r['RGB_sha256'],submitted_mask_SHA256=sha(out/'submitted_masks'/name),prediction_mask_SHA256=sha(out/'prediction_masks'/name),submitted_overlay_SHA256=sha(out/'submitted_labels'/name),prediction_overlay_SHA256=sha(out/'model_predictions'/name),model_SHA256=complete['model_SHA256'],training_epochs=100,selected_epoch=32,prediction_context='crop of native full-field prediction at the submitted ROI coordinates')
 records.append(rec);lookup[pos,t]=name
with (out/'overlay_manifest.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
readme='''# 补充66张图像：根系标记叠加图

全部图像保留500×500原始像素，不缩放、不改变根系标记形状。黄色仅表示根系，新生、既有和T1基线均合为根系。

submitted_labels：你上传JSON的根系标记直接叠加到对应原始RGB图像。
model_predictions：固定完成100轮的新版V35，按内部验证Dice选择第32轮权重，在完整原始视野预测后，按相同坐标提取500px窗口并叠加。
submitted_masks：上传二分类标记，0=背景、255=根系、128=忽略。
prediction_masks：新版二分类预测，0=背景、255=根系，阈值0.5。

左右两种叠加图分开保存，文件名与原500px图像一致。overlay_manifest.csv记录坐标、标记像素、模型和图像SHA256。

66张图像包括52张本轮训练图像与14张本轮内部验证图像；这些对照图不属于独立验证。18页PDF每个位置一页，按T1–T4排列，缺失观测留空注明。PDF嵌入原500px图像，无有损压缩。所有完整原视野预测另附18个位置的数据包。
'''
(out/'README.md').write_text(readme)
doc=fitz.open();times='/tmp/v66-fonts/Times.TTF';bold='/tmp/v66-fonts/Timesbd.TTF'
for pos in sorted({r['position'] for r in rows}):
 page=doc.new_page(width=595.276,height=841.89);page.insert_font(fontname='TNR',fontfile=times);page.insert_font(fontname='TNRB',fontfile=bold)
 page.insert_text((35,29),'V35 root/background segmentation | '+pos,fontname='TNRB',fontsize=15,color=(0,0,0))
 material=next(r['material'] for r in rows if r['position']==pos);page.insert_text((35,47),'Material '+material+' | Native 500 x 500 px crops',fontname='TNR',fontsize=10,color=(0,0,0))
 page.insert_text((99,66),'Submitted root labels',fontname='TNRB',fontsize=11,color=(0,0,0));page.insert_text((343,66),'V35 model prediction',fontname='TNRB',fontsize=11,color=(0,0,0))
 for i,t in enumerate(['T1','T2','T3','T4']):
  yy=78+i*182;page.insert_text((40,yy+88),t,fontname='TNRB',fontsize=13,color=(0,0,0))
  for xx,folder in [(95,'submitted_labels'),(340,'model_predictions')]:
   rect=fitz.Rect(xx,yy,xx+170,yy+170)
   if (pos,t) in lookup:page.insert_image(rect,filename=str(out/folder/lookup[pos,t]))
   else:page.draw_rect(rect,color=(.75,.75,.75),width=.4);page.insert_textbox(rect,'\n\n\n\nNo observation',fontname='TNR',fontsize=12,align=1,color=(0,0,0))
 page.insert_text((35,823),'Yellow: root | 100 training epochs; selected epoch 32 | Prediction threshold: 0.5',fontname='TNR',fontsize=8,color=(0,0,0))
pdf=pub/'V35_66_submitted_and_prediction_overlays.pdf';doc.save(pdf,garbage=4,deflate=True);doc.close()
with fitz.open(pdf) as d:
 assert len(d)==18
 for page in d:
  for img in page.get_images():assert img[2:4]==(500,500)
 # Representative PDF page exported for browser preview only.
 d[0].get_pixmap(matrix=fitz.Matrix(1.3,1.3),alpha=False).save(str(pub/'V35_66_overlay_preview.png'))
zip_path=pub/'V35_66_root_overlay_PNG.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(out.rglob('*')):
  if p.is_file():z.write(p,str(p.relative_to(out)))
assert zip_path.stat().st_size<100*1024*1024 and pdf.stat().st_size<100*1024*1024
assert len(list((out/'submitted_labels').glob('*.png')))==len(list((out/'model_predictions').glob('*.png')))==66
with zipfile.ZipFile(zip_path) as z:assert z.testzip() is None
(pub/'V35_66_overlay_README.md').write_text(readme)
(out/'completion.json').write_text(json.dumps(dict(images=66,positions=18,pages=18,native_size=[500,500],source_masks_unchanged=True,model_prediction_context='same ROI cropped from full-field prediction',pdf_bytes=pdf.stat().st_size,png_zip_bytes=zip_path.stat().st_size),indent=2)+'\n')
print('OVERLAY_COMPLETE',pdf.stat().st_size,zip_path.stat().st_size,flush=True)
