import csv,hashlib,io,json,os,shutil,zipfile
from pathlib import Path
import fitz,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path('/workspace/Temporal-segmentation-model');out=ROOT/'work/V90_all_matched_native';pub=Path('/workspace/shared/downloads');prefix='AllMatched_native_';done=json.load(open(out/'completion.json'));assert done['images']==66 and done['positions']==18
rows=list(csv.DictReader(open(out/'prediction_manifest.csv',encoding='utf-8-sig')));positions=sorted({r['position'] for r in rows});sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old_commit=(ROOT/'work/V90_full250_common66/published_commit.txt').read_text().strip();oldbase=f'https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/{old_commit}/publication/TRGD-full250-common66/'
paired=fitz.open();entries=[]
for pos in positions:
 rr=sorted([r for r in rows if r['position']==pos],key=lambda r:r['period']);folder=out/'predictions'/pos
 zpath=pub/(prefix+pos+'_V35.zip')
 with zipfile.ZipFile(zpath,'w',zipfile.ZIP_STORED) as z:
  for r in rr:
   t=r['period']
   for suffix in ['_RGB.png','_V35_mask.png','_V35_overlay.png','_metadata.json']:z.write(folder/(t+suffix),pos+'/'+t+suffix)
  z.write(folder/'temporal_summary.csv',pos+'/temporal_summary.csv')
 assert zpath.stat().st_size<100*1048576
 with zipfile.ZipFile(zpath) as z:assert z.testzip() is None
 for r in rr:
  t=r['period'];w,h=int(r['width']),int(r['height']);panelh=350*h/w;page=paired.new_page(width=841.89,height=145+panelh);page.insert_font(fontname='TNR',fontfile='/tmp/v66-fonts/Times.TTF');page.insert_font(fontname='TNRB',fontfile='/tmp/v66-fonts/Timesbd.TTF');page.insert_text((35,28),pos+' | '+t+' | Same complete matched RGB',fontname='TNRB',fontsize=16,color=(0,0,0));page.insert_text((35,51),'V35: root / background',fontname='TNRB',fontsize=12,color=(0,0,0));page.insert_text((440,51),'V90: temporal labels on fixed V35 roots',fontname='TNRB',fontsize=12,color=(0,0,0))
  for x,suffix in [(35,'_V35_overlay.png'),(440,'_V90_overlay.png')]:
   im=Image.open(folder/(t+suffix)).convert('RGB');im.thumbnail((1200,1200),Image.Resampling.LANCZOS);b=io.BytesIO();im.save(b,format='JPEG',quality=94,subsampling=0);page.insert_image(fitz.Rect(x,65,x+350,65+panelh),stream=b.getvalue())
  page.insert_text((35,92+panelh),f'Input {w} x {h} px | Foreground difference: 0 pixels | Preview; native PNGs in ZIP files',fontname='TNR',fontsize=10,color=(0,0,0))
  page.insert_text((35,114+panelh),'V90: orange = new root; blue = existing root | T1: existing-root reference',fontname='TNR',fontsize=10,color=(0,0,0))
 entries.append(dict(position=pos,periods=','.join(r['period'] for r in rr),width=int(rr[0]['width']),height=int(rr[0]['height']),V35_zip=zpath.name,V35_bytes=zpath.stat().st_size,V90_zip=oldbase+'TRGD_full250_'+pos+'.zip',V90_native_PDF=oldbase+'TRGD_full250_'+pos+'.pdf'))
 print('EXPORTED_NATIVE_V35',pos,len(rr),zpath.stat().st_size,flush=True)
assert len(paired)==66;paired.save(pub/(prefix+'V35_V90_comparison_preview.pdf'),garbage=4,deflate=True);paired.close();assert (pub/(prefix+'V35_V90_comparison_preview.pdf')).stat().st_size<100*1048576
for src,dst in [('TRGD_full250_all_predictions_overview.pdf',prefix+'V90_overview.pdf'),('TRGD_full250_all_two_class_masks.zip',prefix+'V90_two_class_masks.zip')]:
 target=pub/dst
 if target.exists():assert sha(target)==sha(pub/src)
 else:os.link(pub/src,target)
with zipfile.ZipFile(pub/(prefix+'V35_binary_masks.zip'),'w',zipfile.ZIP_DEFLATED) as z:
 for r in rows:
  pos,t=r['position'],r['period'];z.write(out/'predictions'/pos/(t+'_V35_mask.png'),pos+'/'+t+'_root_mask.png')
 for name in ['prediction_manifest.csv','completion.json','missing_periods.csv']:z.write(out/name,name)
# Full-field same-color check; resizing only this visual preview, not any saved native mask.
pos='0353-1-B3';folder=out/'predictions'/pos;rgb=np.array(Image.open(folder/'T1_RGB.png').convert('RGB'));m=np.array(Image.open(folder/'T1_V35_mask.png'))>0;cl=np.array(Image.open(folder/'T1_V90_classes.png'));assert np.array_equal(m,cl>0)
a=rgb.copy();a[m]=(rgb[m]*.25+np.array([255,193,7])*.75).astype('uint8');b=rgb.copy();b[cl>0]=(rgb[cl>0]*.25+np.array([255,193,7])*.75).astype('uint8');assert np.array_equal(a,b)
im=Image.fromarray(a);im.thumbnail((1050,1050),Image.Resampling.LANCZOS);cw,ch=im.size;canvas=Image.new('RGB',(2*cw+75,ch+100),'white');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('/tmp/v66-fonts/Times.TTF',22);bold=ImageFont.truetype('/tmp/v66-fonts/Timesbd.TTF',24)
draw.text((25,12),f'Complete matched input | {rgb.shape[1]} x {rgb.shape[0]} px | Same foreground: {int(m.sum())} pixels',font=bold,fill='black')
for x,title in [(25,'V35 root foreground'),(50+cw,'V90 foreground, same color')]:draw.text((x,48),title,font=font,fill='black');canvas.paste(im,(x,80))
canvas.save(pub/(prefix+'same_foreground_preview.png'))
readme='''# 全部已匹配共有视野：完整原尺寸图像，固定V35根系前景

本版范围是18个位置、66张完整共有视野图像，保留原尺寸、原范围和6个缺期空位。输入为已有匹配好的RGB，本次不重新配准、裁图或提取视野。

使用已完成100轮训练后选定的第91轮V35，只分根系/背景。V90读取同一张图和同一份V35前景，只赋既有/新生时序类别；所有66张逐像素核验根系范围差异0，输入RGB哈希、根系mask哈希及模型权重哈希均一致。

经过审计，此批完整输入的同权重V35/V90推理已经完成，因此复用已验证的预测缓存，无需重复相同神经网络前向。不是把500px示例放大或拼贴到完整图像。每张记录标明原尺寸和缓存来源，旧权重、原图、先前结果均保留。

V35和V90在本版的对比始终使用同一完整输入。不同输入范围（独立500px图 vs 完整原图）的预测并不保证相同；本版不混用这两种推理口径。V90时序仍使用现有预训练迁移接口，预处理差异保留于来源信息，没有赋予未经验证的时序准确率。

总览和同图对比PDF是浏览版；各位置ZIP内PNG保留原始像素。V90原尺寸PDF链接直接嵌入原尺寸PNG。V35的黄色图为根系/背景，V90橙色新生根、蓝色既有根。T1可见根作为既有参考，缺期留白。

'''
base='https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/All-matched-native/'
readme+=f'- [V90全部时序总览PDF]({base}{prefix}V90_overview.pdf)\n- [V35/V90同图对比PDF（浏览版）]({base}{prefix}V35_V90_comparison_preview.pdf)\n- [全部V35根系/背景mask]({base}{prefix}V35_binary_masks.zip)\n- [全部V90两类mask]({base}{prefix}V90_two_class_masks.zip)\n\n|位置|已有时期|原尺寸|V35原图/根系mask/PNG|V90原图/两类mask/PNG|V90原尺寸PDF|\n|---|---|---|---|---|---|\n'
for r in entries:readme+=f"|{r['position']}|{r['periods']}|{r['width']}×{r['height']}|[ZIP ({r['V35_bytes']/1048576:.1f}MB)]({base}{r['V35_zip']})|[ZIP]({r['V90_zip']})|[PDF]({r['V90_native_PDF']})|\n"
(pub/(prefix+'README.md')).write_text(readme)
for name in ['completion.json','prediction_manifest.csv','missing_periods.csv']:(pub/(prefix+name)).write_bytes((out/name).read_bytes())
files=[p for p in pub.glob(prefix+'*') if p.is_file() and not p.name.endswith('checksums.json')];(pub/(prefix+'checksums.json')).write_text(json.dumps({p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files},indent=2)+'\n')
(out/'export_manifest.json').write_text(json.dumps(dict(entries=entries,total_files=len(files),new_native_V35_bytes=sum(r['V35_bytes'] for r in entries)),indent=2)+'\n');print('ALL_MATCHED_NATIVE_EXPORTS_COMPLETE',len(files),flush=True)
