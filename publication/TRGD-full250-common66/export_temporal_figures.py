import csv,hashlib,io,json,shutil,zipfile
from pathlib import Path
import fitz
import numpy as np
from PIL import Image
ROOT=Path('/workspace/Temporal-segmentation-model');out=ROOT/'work/V90_full250_common66';pub=Path('/workspace/shared/downloads');done=json.load(open(out/'completion.json'));assert done['positions']==18 and done['images']==66
rows=list(csv.DictReader(open(out/'temporal_prediction_manifest.csv',encoding='utf-8-sig')));assert len(rows)==66;positions=sorted({r['position'] for r in rows});assert len(positions)==18
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();latest=ROOT/'work/V35_full250_common66';missing=[]
source_RGB_expected={(v['position'],v['period']):v['source_common_RGB_SHA256'] for v in csv.DictReader(open(latest/'prediction_manifest.csv',encoding='utf-8-sig'))}
for pos in positions:
 for t in ['T1','T2','T3','T4']:
  if not any(r['position']==pos and r['period']==t for r in rows):missing.append((pos,t))
assert len(missing)==6
for r in rows:
 p=out/'predictions'/r['position'];t=r['period'];cl=np.array(Image.open(p/(t+'_classes.png')));mask=np.array(Image.open(latest/r['position']/(t+'_root_mask.png')))>0;rgb=np.array(Image.open(p/(t+'_common_RGB.png')).convert('RGB'));overlay=np.array(Image.open(p/(t+'_root_temporal_overlay.png')).convert('RGB'));domain=np.array(Image.open(p/'common_domain.png'))>0
 assert cl.shape==mask.shape==domain.shape==(int(r['height']),int(r['width']));assert np.isin(cl,[0,1,2]).all() and np.array_equal(cl>0,mask) and not cl[~domain].any();assert np.array_equal(rgb[~mask],overlay[~mask]);assert t!='T1' or not (cl==2).any();assert int((cl==1).sum())+int((cl==2).sum())==int(mask.sum())==int(r['root_pixels']);assert sha(p/(t+'_classes.png'))==r['classes_SHA256'];assert sha(p/(t+'_common_RGB.png'))==r['source_common_RGB_SHA256']==source_RGB_expected[r['position'],t]
 for absent in [t for pp,t in missing if pp==r['position']]:assert not (p/(absent+'_classes.png')).exists()
print('ALL_66_TEMPORAL_FOREGROUND_CHECKS_PASSED',flush=True)
overview=fitz.open();entries=[]
for pos in positions:
 rr=sorted([r for r in rows if r['position']==pos],key=lambda r:r['period']);p=out/'predictions'/pos;w,h=int(rr[0]['width']),int(rr[0]['height']);panelh=350*h/w;pageh=207+2*panelh;native=fitz.open()
 for mode,doc in [('native',native),('preview',overview)]:
  page=doc.new_page(width=841.89,height=pageh);page.insert_font(fontname='TNR',fontfile='/tmp/v66-fonts/Times.TTF');page.insert_font(fontname='TNRB',fontfile='/tmp/v66-fonts/Timesbd.TTF');page.insert_text((35,30),'TRGD (V90) + full250 V35 | '+pos,fontname='TNRB',fontsize=18,color=(0,0,0));page.insert_text((35,49),'Common field only | T1-T4 layout | Missing periods remain blank'+(' | Preview' if mode=='preview' else ''),fontname='TNR',fontsize=10,color=(0,0,0));page.draw_rect(fitz.Rect(35,60,44,69),color=None,fill=(230/255,159/255,0));page.insert_text((50,68),'New root',fontname='TNR',fontsize=10,color=(0,0,0));page.draw_rect(fitz.Rect(128,60,137,69),color=None,fill=(55/255,118/255,185/255));page.insert_text((143,68),'Existing root',fontname='TNR',fontsize=10,color=(0,0,0))
  for i,t in enumerate(['T1','T2','T3','T4']):
   left=35+(i%2)*390;top=94+(i//2)*(panelh+45);rect=fitz.Rect(left,top,left+350,top+panelh);page.insert_text((left,top-7),t,fontname='TNRB',fontsize=14,color=(0,0,0));r=next((r for r in rr if r['period']==t),None)
   if r is None:continue
   image=p/(t+'_root_temporal_overlay.png')
   if mode=='native':page.insert_image(rect,filename=str(image))
   else:
    im=Image.open(image).convert('RGB');im.thumbnail((1500,1500),Image.Resampling.LANCZOS);buffer=io.BytesIO();im.save(buffer,format='JPEG',quality=92,subsampling=0);page.insert_image(rect,stream=buffer.getvalue())
   page.insert_text((left+105,top+panelh+17),str(w)+' x '+str(h)+' px',fontname='TNR',fontsize=10,color=(0,0,0))
  foot='T1: existing-root baseline | Later frames: relative to earlier observed dates | Foreground = updated V35'
  if mode=='preview':foot+=' | Preview'
  page.insert_text((35,pageh-15),foot,fontname='TNR',fontsize=8,color=(0,0,0))
 pdf=pub/f'TRGD_full250_{pos}.pdf';native.save(pdf,garbage=4,deflate=True);native.close();assert pdf.stat().st_size<100*1048576
 with fitz.open(pdf) as d:
  assert len(d[0].get_images())==len(rr) and all(i[2:4]==(w,h) for i in d[0].get_images())
  if pos=='A017-1-B2':d[0].get_pixmap(matrix=fitz.Matrix(1.1,1.1),alpha=False).save(str(pub/'TRGD_full250_preview.png'))
 files=[]
 for r in rr:
  for suffix in ['_classes.png','_root_temporal_overlay.png','_V35_root_mask.png','_common_RGB.png','_ownership.npz','_metadata.json']:files.append(p/(r['period']+suffix))
 files += [p/f for f in ['common_domain.png','completion.json','temporal_summary.csv','preprocessing_provenance.json']]
 zip_path=pub/f'TRGD_full250_{pos}.zip'
 with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_STORED) as z:
  for f in files:z.write(f,pos+'/'+f.name)
 assert zip_path.stat().st_size<100*1048576
 with zipfile.ZipFile(zip_path) as z:assert z.testzip() is None
 entries.append(dict(position=pos,images=len(rr),periods=','.join(r['period'] for r in rr),zip=zip_path.name,pdf=pdf.name,zip_bytes=zip_path.stat().st_size,pdf_bytes=pdf.stat().st_size));print('TEMPORAL_EXPORTED',pos,len(rr),flush=True)
assert len(overview)==18 and sum(len(p.get_images()) for p in overview)==66;overview.save(pub/'TRGD_full250_all_predictions_overview.pdf',garbage=4,deflate=True);overview.close();assert (pub/'TRGD_full250_all_predictions_overview.pdf').stat().st_size<100*1048576
with zipfile.ZipFile(pub/'TRGD_full250_all_two_class_masks.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for r in rows:
  pos,t=r['position'],r['period'];z.write(out/'predictions'/pos/(t+'_classes.png'),pos+'/'+t+'_classes.png')
 for f in ['temporal_prediction_manifest.csv','missing_periods.csv','completion.json']:z.write(out/f,f)
url='https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-full250-common66/'
text='''# TRGD（V90）：使用最新版V35前景预测两类根系时序

V90的根系/背景输入已替换为310源图、1240个250px裁块、补充66张全部纳入训练后固定100轮训练后选出的新版V35（第91轮）概率与前景。重新计算V66时序先验、图结构以及V90 ConvLSTM/GAT、Axis根轴约束、Junction交叉连接约束；V66/V90时序权重保持冻结，没有使用新增根系/背景标记去虚构时序训练目标。

最终输出仅为0=背景、1=既有根、2=新生根。延伸与持续归为既有根。T1所有可见根作为既有根基线；后续类别比较此前真实观测，跨缺失期不补造中间影像。只处理18个共有视野、66张真实图像。PDF固定T1–T4四格，6个缺期留空。

所有时序输出的前景均与新版V35 mask逐像素相等，新增背景根系0像素，移除V35根系0像素。共有区域外没有标记，非根系RGB保持不变。该检查证明前景输入绑定一致，不代表时序类别的独立准确率。

本次沿用现有V90预训练迁移接口。它使用重建的骨架节点分区/邻接和一个最近交叉点度数几何代理，未包含原流程未发布的稠密局部修正预处理，因此不称为原张量绑定流程的完全数值复现，也不沿用旧位置的准确率作为本次成绩。原始V35与V90权重保留。

'''
text+=f'- [全部两类时序预测总览PDF（浏览版，缺期留空）]({url}TRGD_full250_all_predictions_overview.pdf)\n- [全部66张两类mask]({url}TRGD_full250_all_two_class_masks.zip)\n- [更新后的组合模型与运行代码]({url}TRGD_full250_model_runtime.zip)\n\n各位置ZIP包括共有RGB、两类mask、标记叠加PNG、V35前景、轴/轨迹推断编号与来源元数据。原尺寸PDF嵌入原像素PNG；浏览总览缩小显示以便查看，原始像素在各位置下载包中。轨迹编号是网络及约束推断的关联，不作为另一个根系语义类别。\n\n| 位置 | 实际时期 | 原尺寸PNG和mask | 原尺寸PDF |\n|---|---|---|---|\n'
for r in entries:text+=f"| {r['position']} | {r['periods']} | [ZIP ({r['zip_bytes']/1048576:.1f} MB)]({url}{r['zip']}) | [PDF ({r['pdf_bytes']/1048576:.1f} MB)]({url}{r['pdf']}) |\n"
text+='\n权重SHA256：\n'+''.join('- '+k+': `'+v+'`\n' for k,v in done['weights_SHA256'].items());(pub/'TRGD_full250_README.md').write_text(text)
for f in ['temporal_prediction_manifest.csv','missing_periods.csv','completion.json']:shutil.copy2(out/f,pub/('TRGD_full250_'+f))
verification=dict(positions=18,images=66,classes={'0':'background','1':'existing_root','2':'new_root'},foreground_matches_latest_V35_pixelwise=True,added_root_pixels=0,removed_root_pixels=0,missing_frames_synthesized=False,missing_panels=6,temporal_weights_unchanged=True,method='existing V90 pretrained transfer interface',weights_SHA256=done['weights_SHA256']);(pub/'TRGD_full250_verification.json').write_text(json.dumps(verification,indent=2)+'\n');(out/'export_manifest.json').write_text(json.dumps(dict(entries=entries,images=66,positions=18),indent=2)+'\n');print('TEMPORAL_EXPORTS_READY',flush=True)
