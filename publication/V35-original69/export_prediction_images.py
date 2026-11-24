import csv,hashlib,json,shutil,zipfile
from pathlib import Path
import fitz
ROOT=Path('/workspace/Temporal-segmentation-model');out=ROOT/'work/V35_all_original69';pub=Path('/workspace/shared/downloads');done=json.load(open(out/'completion.json'));rows=list(csv.DictReader(open(out/'prediction_manifest.csv',encoding='utf-8-sig')));assert done['images']==len(rows)==69;positions=sorted({r['position'] for r in rows});assert len(positions)==19
archives=[];pdfs=[]
for pos in positions:
 rr=sorted([r for r in rows if r['position']==pos],key=lambda r:r['period']);pdf=pub/f'V35_original69_{pos}.pdf';doc=fitz.open();page=doc.new_page(width=841.89,height=841.89);page.insert_font(fontname='TNR',fontfile='/tmp/v66-fonts/Times.TTF');page.insert_font(fontname='TNRB',fontfile='/tmp/v66-fonts/Timesbd.TTF')
 page.insert_text((35,35),'V35 root/background prediction | '+pos,fontname='TNRB',fontsize=19,color=(0,0,0));page.insert_text((35,57),'Complete native original images | All root categories merged | No registration or resizing',fontname='TNR',fontsize=11,color=(0,0,0))
 for i,t in enumerate(['T1','T2','T3','T4']):
  left=35+(i%2)*390;top=90+(i//2)*350;rect=fitz.Rect(left,top,left+350,top+312);page.insert_text((left,top-8),t,fontname='TNRB',fontsize=15,color=(0,0,0));r=next((r for r in rr if r['period']==t),None)
  if r:
   # Inserting PNG directly preserves native pixel arrays and avoids lossy resampling.
   page.insert_image(rect,filename=str(out/pos/(t+'_root_overlay.png')),keep_proportion=True);page.insert_text((left+110,top+330),r['width']+' x '+r['height']+' px',fontname='TNR',fontsize=11,color=(0,0,0))
  else:page.draw_rect(rect,color=(.8,.8,.8),width=.5);page.insert_textbox(rect,'\n\n\n\n\n\n\nNo original observation',fontname='TNR',fontsize=12,align=1,color=(0,0,0))
 page.insert_text((35,823),'Yellow: predicted root | Fixed 100-epoch run; validation-selected epoch 32 | Threshold 0.5',fontname='TNR',fontsize=10,color=(0,0,0));doc.save(pdf,garbage=4,deflate=True);doc.close();assert pdf.stat().st_size<100*1048576
 with fitz.open(pdf) as d:
  imgs=d[0].get_images();assert len(imgs)==len(rr);assert sorted((x[2],x[3]) for x in imgs)==sorted((int(r['width']),int(r['height'])) for r in rr)
  if pos=='0353-1-B3':d[0].get_pixmap(matrix=fitz.Matrix(.85,.85),alpha=False).save(str(pub/'V35_original69_preview.png'))
 pdfs.append(dict(position=pos,file=pdf.name,bytes=pdf.stat().st_size))
 # Pack whole native images without splitting pixels or changing file dimensions.
 groups=[];current=[];size=0
 for r in rr:
  batch=[out/pos/(r['period']+suffix) for suffix in ['_root_mask.png','_root_overlay.png','_metadata.json']];bsize=sum(p.stat().st_size for p in batch)
  assert bsize<90*1048576
  if current and size+bsize>90*1048576:groups.append(current);current=[];size=0
  current.extend(batch);size+=bsize
 if current:groups.append(current)
 for i,group in enumerate(groups,1):
  name=f'V35_original69_{pos}'+('' if len(groups)==1 else f'_part{i}')+'.zip';dest=pub/name
  with zipfile.ZipFile(dest,'w',zipfile.ZIP_STORED) as z:
   for p in group:z.write(p,pos+'/'+p.name)
  assert dest.stat().st_size<100*1048576
  with zipfile.ZipFile(dest) as z:assert z.testzip() is None
  archives.append(dict(position=pos,file=name,bytes=dest.stat().st_size,periods=','.join(r['period'] for r in rr)))
 print('EXPORTED',pos,'pdf_bytes',pdf.stat().st_size,'zip_parts',len(groups),flush=True)
with zipfile.ZipFile(pub/'V35_original69_all_binary_masks.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for r in rows:z.write(out/r['position']/(r['period']+'_root_mask.png'),r['position']+'/'+r['period']+'_root_mask.png')
 for f in ['prediction_manifest.csv','missing_observations.csv','completion.json']:z.write(out/f,f)
rooturl='https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-original69/'
text='''# 新V35：全部69张原始图像预测

本次覆盖源档案中19个观测位置、9个物理根管、69张真实原始输入图像，包含3个种质及可用T1–T4。与此前66张共有视野预测分开保存。没有把共有视野当作完整原始图像；未完成定位的Q1-3-Z3也逐张做了根系/背景分割。

使用合并310张标记、重新划分248训练/62内部验证后固定完成100轮训练的V35；按内部验证Dice选择第32轮权重，阈值0.5。原权重保留。这里只做根系/背景二分类，不区分新生与既有根。

每张输出宽高与原始输入一致，不裁剪、不缩放、不配准，不做形态学清理或人工修改。原始JPG/PNG文件逐个核对源SHA256，预测PNG逐个核对原始宽高。黄色代表模型预测根系，其余RGB像素不变。PDF直接嵌入同一原尺寸PNG，不做有损重采样。

每个位置ZIP包括根系二值mask（0=背景，255=根系）、原图上的预测叠加PNG，以及来源、模型、尺寸和输出哈希元数据。完整float32概率在计算输出中保留，元数据记录概率哈希；这里的下载包交付原尺寸mask与叠加图。根系像素比例不是准确率；原始图像没有新增独立真值，本次没有独立验证。

'''
text+=f'- [全部69张二值mask一次下载]({rooturl}V35_original69_all_binary_masks.zip)\n\n'
text+='| 种质 | 观测位置 | 实际时期 | 原尺寸PNG和mask | 原尺寸PDF |\n|---|---|---|---|---|\n'
for pos in positions:
 rr=sorted([r for r in rows if r['position']==pos],key=lambda r:r['period']);aa=[r for r in archives if r['position']==pos];pp=next(r for r in pdfs if r['position']==pos);links=' / '.join(f"[ZIP{n if len(aa)>1 else ''} ({a['bytes']/1048576:.1f} MB)]({rooturl}{a['file']})" for n,a in enumerate(aa,1));text+=f"| {rr[0]['material']} | {pos} | {','.join(r['period'] for r in rr)} | {links} | [PDF ({pp['bytes']/1048576:.1f} MB)]({rooturl}{pp['file']}) |\n"
text+='\n缺失时点见 missing_observations.csv；不补造图像。各期原始尺寸可能不同，不把页面排版中的并列图像解释为已经配准。\n\n模型SHA256：`'+done['model_SHA256']+'`。源数据仓库Annalee811/LI，固定源版本cd161aeebe942f515ee98c88881991ad02784631，来源记录与每张PNG的模型信息保存在清单中。\n'
(pub/'V35_original69_README.md').write_text(text)
for f in ['prediction_manifest.csv','missing_observations.csv','completion.json']:shutil.copy2(out/f,pub/('V35_original69_'+f))
shutil.copy2('/tmp/LI-all-source_manifest.json',pub/'V35_original69_source_manifest.json')
files=[p for p in pub.glob('V35_original69_*') if p.is_file() and not p.name.endswith('checksums.json')];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();checks={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files};(pub/'V35_original69_checksums.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n');(out/'export_manifest.json').write_text(json.dumps(dict(images=69,positions=19,archives=archives,pdfs=pdfs,total_bytes=sum(p.stat().st_size for p in files)),indent=2)+'\n');print('ALL69_EXPORTED',sum(p.stat().st_size for p in files),flush=True)
