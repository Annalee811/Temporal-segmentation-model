import csv,hashlib,io,json,zipfile,shutil
from pathlib import Path
import fitz
import numpy as np
from PIL import Image
ROOT=Path('/workspace/Temporal-segmentation-model');out=ROOT/'work/V35_four_period_common';pub=Path('/workspace/shared/downloads');rows=list(csv.DictReader(open(out/'prediction_manifest.csv',encoding='utf-8-sig')));done=json.load(open(out/'completion.json'));assert len(rows)==66 and done['positions']==18
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
positions=sorted({r['position'] for r in rows});missing=[]
for pos in positions:
 for t in ['T1','T2','T3','T4']:
  if not any(r['position']==pos and r['period']==t for r in rows):missing.append(dict(position=pos,period=t,display='blank panel; no synthetic image or zero mask'))
assert len(missing)==6
with (out/'missing_periods.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(missing[0]));w.writeheader();w.writerows(missing)
for r in rows:
 folder=out/r['position'];t=r['period'];rgb=np.array(Image.open(folder/(t+'_common_RGB.png')).convert('RGB'));mask=np.array(Image.open(folder/(t+'_root_mask.png')));domain=np.array(Image.open(folder/'common_domain.png'))>0;overlay=np.array(Image.open(folder/(t+'_root_overlay.png')).convert('RGB'));assert rgb.shape[:2]==mask.shape==domain.shape==(int(r['height']),int(r['width']));assert not (mask[~domain]>0).any()
 with np.load(folder/(t+'_root_probability.npz')) as z:assert np.array_equal(z['common_domain']>0,domain) and np.array_equal((z['root_probability']>=.5)&domain,mask==255)
 assert np.array_equal(rgb[mask==0],overlay[mask==0]);assert sha(folder/(t+'_common_RGB.png'))==r['source_common_RGB_SHA256']
 if r['cached_identical_input_and_model_prediction']=='True':
  cache=ROOT/'work/V35_310_all18_predictions'/r['position'];old=json.load(open(cache/(t+'_metadata.json')));assert sha(cache/(t+'_root_probability.npz'))==old['probability_file_SHA256'];assert old['model_SHA256']==r['model_SHA256']
print('ALL66_COMMON_DOMAIN_CHECKS_PASSED',flush=True)
overview=fitz.open();entries=[]
for pos in positions:
 rr=sorted([r for r in rows if r['position']==pos],key=lambda r:r['period']);assert len({(r['width'],r['height']) for r in rr})==1;folder=out/pos;w,h=int(rr[0]['width']),int(rr[0]['height']);panelh=350*h/w;pageh=190+2*panelh;native=fitz.open()
 for mode,doc in [('native',native),('preview',overview)]:
  page=doc.new_page(width=841.89,height=pageh);page.insert_font(fontname='TNR',fontfile='/tmp/v66-fonts/Times.TTF');page.insert_font(fontname='TNRB',fontfile='/tmp/v66-fonts/Timesbd.TTF');page.insert_text((35,30),'V35 common observation field | '+pos,fontname='TNRB',fontsize=18,color=(0,0,0));subtitle='T1-T4 layout | Common field of observed periods | Blank panels: missing observations'
  if mode=='preview':subtitle+=' | Preview'
  page.insert_text((35,49),subtitle,fontname='TNR',fontsize=10,color=(0,0,0))
  for i,t in enumerate(['T1','T2','T3','T4']):
   left=35+(i%2)*390;top=78+(i//2)*(panelh+45);rect=fitz.Rect(left,top,left+350,top+panelh);page.insert_text((left,top-7),t,fontname='TNRB',fontsize=14,color=(0,0,0));r=next((r for r in rr if r['period']==t),None)
   if r is None:continue
   path=folder/(t+'_root_overlay.png')
   if mode=='native':page.insert_image(rect,filename=str(path))
   else:
    im=Image.open(path).convert('RGB');im.thumbnail((1500,1500),Image.Resampling.LANCZOS);b=io.BytesIO();im.save(b,format='JPEG',quality=92,subsampling=0);page.insert_image(rect,stream=b.getvalue())
   page.insert_text((left+105,top+panelh+17),str(w)+' x '+str(h)+' px',fontname='TNR',fontsize=10,color=(0,0,0))
  foot='Yellow: root | Common-domain mask applied | 100 training epochs; selected epoch 32 | Threshold 0.5'
  if mode=='preview':foot+=' | Preview'
  page.insert_text((35,pageh-15),foot,fontname='TNR',fontsize=8,color=(0,0,0))
 pdf=pub/f'V35_common_fields_{pos}.pdf';native.save(pdf,garbage=4,deflate=True);native.close();assert pdf.stat().st_size<100*1048576
 with fitz.open(pdf) as d:
  assert len(d[0].get_images())==len(rr) and all(im[2:4]==(w,h) for im in d[0].get_images())
  if pos=='A017-1-B2':d[0].get_pixmap(matrix=fitz.Matrix(1.1,1.1),alpha=False).save(str(pub/'V35_common_fields_preview.png'))
 zipname=f'V35_common_fields_{pos}.zip';path=pub/zipname
 with zipfile.ZipFile(path,'w',zipfile.ZIP_STORED) as z:
  for r in rr:
   for suffix in ['_common_RGB.png','_root_mask.png','_root_overlay.png','_metadata.json']:
    f=folder/(r['period']+suffix);z.write(f,pos+'/'+f.name)
  for f in ['common_domain.png','geometry_metadata.json']:z.write(folder/f,pos+'/'+f)
 assert path.stat().st_size<100*1048576
 with zipfile.ZipFile(path) as z:assert z.testzip() is None
 entries.append(dict(position=pos,material=rr[0]['material'],width=w,height=h,images=len(rr),observed_periods=','.join(r['period'] for r in rr),zip=zipname,zip_bytes=path.stat().st_size,pdf=pdf.name,pdf_bytes=pdf.stat().st_size));print('COMMON_EXPORTED',pos,len(rr),flush=True)
assert len(overview)==18 and sum(len(page.get_images()) for page in overview)==66;overview.save(pub/'V35_common_fields_all_predictions_overview.pdf',garbage=4,deflate=True);overview.close();assert (pub/'V35_common_fields_all_predictions_overview.pdf').stat().st_size<100*1048576
with zipfile.ZipFile(pub/'V35_common_fields_all_binary_masks.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for r in rows:z.write(out/r['position']/(r['period']+'_root_mask.png'),'root_masks/'+r['position']+'/'+r['period']+'_root_mask.png')
 for pos in positions:z.write(out/pos/'common_domain.png','common_domains/'+pos+'.png')
 for f in ['prediction_manifest.csv','missing_periods.csv','excluded_positions.csv','completion.json']:z.write(out/f,f)
url='https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-common-fields/'
text='''# 新版V35：只保留共有视野，缺失期留空

保留18个已有共有视野的位置、66张真实共有视野图像，固定按T1–T4排版。6个缺失时点在PDF中只留空白，不生成图像、根系标记或全零mask来替代缺失数据。各位置已观测时期的RGB与根系标记使用同一共有范围、同一尺寸。没有共有定位区域的Q1-3-Z3单列于excluded_positions.csv，不用完整原图替代共有视野。

共有域沿用已保存的区域坐标、采样和T1参考尺度，本次不重新配准或额外缩放。共有域掩膜限制根系输出，分隔行不计入视野、不标根系。缺一期的位置使用已观测期间的共有视野，不宣称存在缺失时期的共有像素。

模型为合并310张标记、重新划分248训练/62内部验证后固定训练100轮的V35，内部验证选定第32轮，阈值0.5；只分背景与根系，所有时序根类别均归为根系，原权重保留。

各位置ZIP含实际存在的共有RGB、原尺寸叠加PNG、二值root mask、共有域及来源信息。原尺寸PDF直接嵌入相同PNG，无有损缩放。浏览总览为了便于一次查看而缩小图像，原始像素版本见各位置PNG/PDF。

'''
text+=f'- [全部共有视野预测总览PDF（18页，缺期留空）]({url}V35_common_fields_all_predictions_overview.pdf)\n- [全部根系mask与共有域]({url}V35_common_fields_all_binary_masks.zip)\n\n| 种质 | 位置 | 实际已有时期 | 共有视野尺寸 | 原尺寸RGB、PNG和mask | 原尺寸PDF |\n|---|---|---|---|---|---|\n'
for r in entries:text+=f"| {r['material']} | {r['position']} | {r['observed_periods']} | {r['width']} x {r['height']} | [ZIP ({r['zip_bytes']/1048576:.1f} MB)]({url}{r['zip']}) | [PDF ({r['pdf_bytes']/1048576:.1f} MB)]({url}{r['pdf']}) |\n"
text+='\n模型SHA256：`'+done['model_SHA256']+'`。共有视野来源：Annalee811/LI固定版本cd161aeebe942f515ee98c88881991ad02784631。模型与输入哈希相同的既有预测经重新核对后复用，其余重新推理。\n';(pub/'V35_common_fields_README.md').write_text(text)
for f in ['prediction_manifest.csv','missing_periods.csv','excluded_positions.csv','completion.json']:shutil.copy2(out/f,pub/('V35_common_fields_'+f))
v=dict(images=66,positions=18,period_slots=['T1','T2','T3','T4'],missing_panels=6,missing_not_synthesized=True,identical_common_size_per_position=True,zero_roots_outside_common_domain=True,source_common_RGB_unchanged=True,nonroot_background_unchanged=True,model_SHA256=done['model_SHA256']);(pub/'V35_common_fields_verification.json').write_text(json.dumps(v,indent=2)+'\n');files=[p for p in pub.glob('V35_common_fields_*') if p.is_file() and not p.name.endswith('checksums.json')];(pub/'V35_common_fields_checksums.json').write_text(json.dumps({p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files},indent=2)+'\n');(out/'available_export_manifest.json').write_text(json.dumps(dict(entries=entries,bytes=sum(p.stat().st_size for p in files)),indent=2)+'\n');print('ALL_COMMON_AVAILABLE_EXPORTS_COMPLETE',sum(p.stat().st_size for p in files),flush=True)
