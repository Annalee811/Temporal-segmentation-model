import csv,hashlib,json,shutil,zipfile
from pathlib import Path
root=Path('/workspace/Temporal-segmentation-model');out=root/'work/V35_310_all18_predictions';pub=Path('/workspace/shared/downloads');done=json.load(open(out/'completion.json'));records=list(csv.DictReader(open(out/'prediction_manifest.csv',encoding='utf-8-sig')));positions=sorted({r['position'] for r in records});assert len(records)==66 and len(positions)==18
entries=[]
for pos in positions:
 file=pub/f'V35_all18_{pos}.zip'
 with zipfile.ZipFile(file,'w',zipfile.ZIP_STORED) as z:
  for p in sorted((out/pos).glob('*')):
   if p.is_file():z.write(p,pos+'/'+p.name)
 assert file.stat().st_size<100*1024*1024,(pos,file.stat().st_size)
 with zipfile.ZipFile(file) as z:assert z.testzip() is None
 rr=[r for r in records if r['position']==pos];entries.append(dict(position=pos,material=rr[0]['material'],tube=rr[0]['tube'],periods=','.join(r['period'] for r in rr),images=len(rr),zip_file=file.name,bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
 print('ARCHIVED',pos,file.stat().st_size,flush=True)
with zipfile.ZipFile(pub/'V35_all18_binary_masks.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for pos in positions:
  for p in (out/pos).glob('*_root_mask.png'):z.write(p,pos+'/'+p.name)
 for f in ['prediction_manifest.csv','position_summary.csv','missing_observations.csv','completion.json']:z.write(out/f,f)
for f in ['prediction_manifest.csv','position_summary.csv','missing_observations.csv','completion.json']:
 shutil.copy2(out/f,pub/('V35_all18_'+f))
modelcommit=(root/'work/V35_310_100epochs/published_commit.txt').read_text().strip();intro=f'''# 新版 V35：全部18个观测位置的原始视野预测

模型已将原244张和新增66张二分类标记合并重新划分为248训练/62内部验证，并完整训练100轮。从原V35第77轮权重继续训练，内部验证选中第32轮权重，SHA256为 `{done['model_SHA256']}`。原V35权重保持不变。本轮不包含独立验证。

- [新版模型与原权重备份](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/{modelcommit}/publication/V35-310-100epochs/V35_310_100epochs_best_model.zip)
- [实际100轮训练曲线PDF](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/{modelcommit}/publication/V35-310-100epochs/V35_310_100epochs_training.pdf)
- [全部66张二值根系mask快速下载](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-all18-predictions/V35_all18_binary_masks.zip)

以下为18个位置、66张现有完整视野图像的真实网络预测。按位置分包，保留输入原始宽高，无缩放、形态学清理或人工后处理。未生成缺失时点图像。

每个包包含：
- `Tn_root_mask.png`：0=背景，255=根系，所有根系统一一类，阈值0.5。
- `Tn_root_probability.npz`：原分辨率float32概率，`numpy.load(path)['root_probability']`读取，未量化。
- `Tn_root_overlay.png`：根系标记叠加在原RGB上，黄色只表示根系。非根像素保持原图RGB不变。
- `Tn_metadata.json`：原图及结果SHA256、宽高、预测根系像素数、模型来源。

`root_area_fraction`为预测根系像素占总像素的比例，不是准确率，也不是生长速率。完整图像没有额外逐像素人工真值，因此此处只交付模型预测，不把它们称为已验证的准确分割。

| 种质 | 观测位置 | 时期 | 完整结果ZIP |
|---|---|---|---|
'''
for r in entries:intro+=f"| {r['material']} | {r['position']} | {r['periods']} | [下载（{r['bytes']/1048576:.1f} MB）](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-all18-predictions/{r['zip_file']}) |\n"
intro+='\n缺失观测：A017-1-B2/T3、A017-1-Z2/T3、Q1-1-B2/T2、Q1-1-Z3/T4、Q1-2-Z2/T2、Q1-2-Z3/T3。\n\n补充的66张500px图像叠加对照另在 V35-66-overlays 文件夹，分别显示上传根系标记与模型预测，二者不混为同一来源。\n'
(pub/'V35_all18_README.md').write_text(intro)
(pub/'V35_all18_download_manifest.json').write_text(json.dumps(dict(model_SHA256=done['model_SHA256'],images=66,positions=18,archives=entries),ensure_ascii=False,indent=2)+'\n')
files=[p for p in pub.glob('V35_all18_*') if p.is_file() and not p.name.endswith('checksums.json')];(pub/'V35_all18_checksums.json').write_text(json.dumps({p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files},indent=2)+'\n');print('ALL18_PACKAGED',flush=True)
