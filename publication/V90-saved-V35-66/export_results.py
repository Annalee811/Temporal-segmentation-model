import csv,hashlib,io,json,zipfile
from pathlib import Path
import fitz,numpy as np
from PIL import Image
ROOT=Path('/workspace/Temporal-segmentation-model');out=ROOT/'work/V90_saved_V35_66';pub=Path('/workspace/shared/downloads');prefix='V90_savedV35_';done=json.load(open(out/'completion.json'));assert done['images']==66 and done['positions']==18
rows=list(csv.DictReader(open(out/'prediction_manifest.csv',encoding='utf-8-sig')));assert len(rows)==66
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();positions=sorted({r['position'] for r in rows});missing=[]
with zipfile.ZipFile(pub/'V35_full250_all66_supplemental_predictions.zip') as original:
 for r in rows:
  d=out/'predictions'/r['position'];t=r['period'];cl=np.array(Image.open(d/(t+'_V90_classes.png')));mask=np.array(Image.open(d/(t+'_V35_mask.png')))>0;rgb=np.array(Image.open(d/(t+'_RGB.png')).convert('RGB'));overlay=np.array(Image.open(d/(t+'_V90_overlay.png')).convert('RGB'));stem=Path(r['input_image']).stem
  assert np.array_equal(cl>0,mask) and np.isin(cl,[0,1,2]).all() and cl.shape==(500,500)
  assert (d/(t+'_V35_mask.png')).read_bytes()==original.read(stem+'_root_mask.png')
  assert (d/(t+'_V35_overlay.png')).read_bytes()==original.read(stem+'_root_overlay.png')
  assert sha(d/(t+'_RGB.png'))==r['RGB_SHA256'];assert np.array_equal(rgb[~mask],overlay[~mask])
  assert t!='T1' or not (cl==2).any()
print('VERIFIED_ALL66_AGAINST_EXACT_USER_SELECTED_ZIP',flush=True)
for pos in positions:
 for t in ['T1','T2','T3','T4']:
  if not any(r['position']==pos and r['period']==t for r in rows):
   missing.append((pos,t));assert not (out/'predictions'/pos/(t+'_V90_classes.png')).exists()
assert len(missing)==6
def fonts(page):
 page.insert_font(fontname='TNR',fontfile='/tmp/v66-fonts/Times.TTF');page.insert_font(fontname='TNRB',fontfile='/tmp/v66-fonts/Timesbd.TTF')
overview=fitz.open();paired=fitz.open()
for pos in positions:
 rr=sorted([r for r in rows if r['position']==pos],key=lambda r:r['period']);d=out/'predictions'/pos;page=overview.new_page(width=841.89,height=900);fonts(page);page.insert_text((35,30),'V90 temporal roots | fixed V35 foreground | '+pos,fontname='TNRB',fontsize=17,color=(0,0,0));page.insert_text((35,50),'Provided registered500px images | No cropping or registration | Missing dates blank',fontname='TNR',fontsize=10,color=(0,0,0))
 for x,color,label in [(35,(230/255,159/255,0),'New root'),(170,(55/255,118/255,185/255),'Existing root')]:
  page.draw_rect(fitz.Rect(x,63,x+10,73),color=None,fill=color);page.insert_text((x+16,72),label,fontname='TNR',fontsize=11,color=(0,0,0))
 for i,t in enumerate(['T1','T2','T3','T4']):
  x=35+(i%2)*390;y=110+(i//2)*390;page.insert_text((x,y-8),t,fontname='TNRB',fontsize=14,color=(0,0,0))
  if not any(r['period']==t for r in rr):continue
  page.insert_image(fitz.Rect(x,y,x+350,y+350),filename=str(d/(t+'_V90_overlay.png')))
  q=paired.new_page(width=841.89,height=485);fonts(q);q.insert_text((35,30),pos+' | '+t+' | Same RGB, same foreground',fontname='TNRB',fontsize=17,color=(0,0,0));q.insert_text((35,53),'V35: saved root/background segmentation',fontname='TNRB',fontsize=12,color=(0,0,0));q.insert_text((440,53),'V90: existing/new root labels only',fontname='TNRB',fontsize=12,color=(0,0,0));q.insert_image(fitz.Rect(35,67,385,417),filename=str(d/(t+'_V35_overlay.png')));q.insert_image(fitz.Rect(440,67,790,417),filename=str(d/(t+'_V90_overlay.png')));q.insert_text((35,447),'500 x 500 px | Foreground difference: 0 pixels | V35 root mask is unchanged',fontname='TNR',fontsize=11,color=(0,0,0))
  if pos=='0353-1-B3' and t=='T1':q.get_pixmap(matrix=fitz.Matrix(1.4,1.4),alpha=False).save(str(pub/(prefix+'same_foreground_preview.png')))
 page.insert_text((35,875),'T1: existing-root reference | V35 mask unchanged | 500 x 500 px | Text: Times New Roman',fontname='TNR',fontsize=10,color=(0,0,0))
assert len(overview)==18 and sum(len(p.get_images()) for p in overview)==66
assert len(paired)==66
overview.save(pub/(prefix+'temporal_overview.pdf'),garbage=4,deflate=True);paired.save(pub/(prefix+'V35_V90_comparison.pdf'),garbage=4,deflate=True)
for doc in [overview,paired]:doc.close()
files=[(p,str(p.relative_to(out/'predictions'))) for p in (out/'predictions').rglob('*') if p.is_file()]
with zipfile.ZipFile(pub/(prefix+'all66_images_masks.zip'),'w',zipfile.ZIP_DEFLATED) as z:
 for p,name in files:z.write(p,name)
 for name in ['prediction_manifest.csv','completion.json','missing_periods.csv']:z.write(out/name,name)
with zipfile.ZipFile(pub/(prefix+'two_class_masks.zip'),'w',zipfile.ZIP_DEFLATED) as z:
 for r in rows:
  pos,t=r['position'],r['period'];z.write(out/'predictions'/pos/(t+'_V90_classes.png'),pos+'/'+t+'_classes.png')
 z.write(out/'completion.json','completion.json')
readme='''# 固定你选定的V35根系结果，V90只判断根系时序

本版直接使用你指定的V35_full250_all66_supplemental_predictions.zip内66张模型预测root_mask和其原RGB（每张500×500）。没有重新配准、重新裁图、重建视野或缩放。

V35是唯一的根系/背景来源；V90不另行分割根系背景，仅对已有根系像素预测1既有根、2新生根，0为背景。T1全部可见根为既有根参考，延伸和持续合并为既有。18个位置、66张真实输入，6个缺失日期保留空位。位置ID按这批输入保留，包括Q1-1-Z2。

已将每张输出的V35mask与原ZIP成员逐字节比较，并确认每张V90掩膜的非背景区域与之逐像素相同。66张均0像素差异。原V35黄色叠加图也逐字节保留。第一张0353-1-B3/T1沿用原3764个根系像素，V90的根系像素仍为3764。

为计算冻结V66时序先验，按完全相同的500px输入和V35权重恢复软概率，并逐张断言0.5阈值与指定掩膜完全相同；这一步不改变指定掩膜。随后执行冻结V90 ConvLSTM/GAT、Axis和Junction，仅赋根系时序类别。模型权重未改变，没有使用提交标记覆盖预测。

时序沿用现有预训练迁移接口：骨架分区/邻接为重建接口，一个几何特征用最近交叉点度数代理，原未发布稠密局部修正预处理未包含；差异保留在来源说明。本版没有给出未经验证的时序准确率。

文件：temporal_overview.pdf为18页T1–T4时序图；V35_V90_comparison.pdf为66页同图同前景对比；all66_images_masks.zip包含原RGB、原V35黄色预测、原V35mask、V90时序PNG和mask、元数据；two_class_masks.zip仅机器掩膜。所有图像500px原尺寸，PDF黑字Times New Roman。
'''
(pub/(prefix+'README.md')).write_text(readme)
for name in ['completion.json','prediction_manifest.csv','missing_periods.csv']:(pub/(prefix+name)).write_bytes((out/name).read_bytes())
verification=dict(positions=18,images=66,input_size=[500,500],mask_source_zip_SHA256=done['source_V35_zip_SHA256'],all66_V35_masks_byte_identical_to_user_selected_archive=True,all66_V90_foregrounds_pixel_identical_to_selected_V35=True,foreground_difference_pixels=0,RGB_unchanged=True,re_registered=False,re_cropped=False,resized=False,missing_dates_synthesized=False)
(pub/(prefix+'verification.json')).write_text(json.dumps(verification,indent=2)+'\n')
for p in pub.glob(prefix+'*'):assert p.stat().st_size<100*1048576,(p,p.stat().st_size)
checks={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in pub.glob(prefix+'*') if p.is_file() and not p.name.endswith('checksums.json')};(pub/(prefix+'checksums.json')).write_text(json.dumps(checks,indent=2)+'\n');print('SAVED_V35_EXPORTS_READY',len(checks),flush=True)
