import csv,hashlib,json,zipfile
from pathlib import Path
root=Path('/workspace/Temporal-segmentation-model');out=root/'work/V90_saved_V35_66';pub=Path('/workspace/shared/downloads');prefix='V90_savedV35_'
rows=[r for r in csv.DictReader(open(root/'work/V35_full250_all66/data/parents.csv',encoding='utf-8-sig')) if r['source']=='submitted_66'];assert len(rows)==66
minimal=[{k:r[k] for k in ['image_id','source','position','period','RGB_sha256']} for r in rows]
parent=out/'input_parents.csv'
with parent.open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(minimal[0]));w.writeheader();w.writerows(minimal)
original=pub/'V35_full250_all66_supplemental_predictions.zip'
with zipfile.ZipFile(pub/(prefix+'inputs.zip'),'w',zipfile.ZIP_DEFLATED) as z:
 for r in rows:z.write(root/'work/V35_310_100epochs/data/RGB'/r['image_id'],'inputs/RGB/'+r['image_id'])
 z.write(original,'inputs/V35_selected_predictions.zip');z.write(parent,'inputs/parents.csv')
old=pub/'TRGD_full250_model_runtime.zip';inventory={}
readme='''# 同图同V35掩膜的V90时序流程

V35是唯一的根系/背景来源。输入为已经匹配的66张500×500原RGB和用户选定的V35预测mask。V90只预测现有根系像素的新生/既有类别。不执行配准、裁图、视野提取或缩放。

把本模型包与V90_savedV35_inputs.zip解压至同一目录，然后：

```bash
python -m pip install -r requirements.txt
python 04_ConvLSTM_GAT_V90/predict_saved_v35.py --masks-zip inputs/V35_selected_predictions.zip --parents inputs/parents.csv --RGB-root inputs/RGB --runtime runtime --output results
```

V35已有mask是不可改变的前景。为了V66软特征，按同一500px输入、同一第91轮V35恢复概率并断言其0.5阈值逐像素复现mask；V90阶段使用缓存的概率，只运行时序先验、ConvLSTM/GAT、Axis、Junction。任何根系前景差异都会停止输出。

原V35预测ZIP包含原提交标记文件，但本程序只读root_mask与root_overlay，不读取submitted_mask。缺期不生成图像；T1可见根作为既有根参考。延伸/持续合并为既有，输出0背景、1既有、2新生。

三份模型权重与之前一致，没有额外训练。现有V90迁移接口的骨架分区/邻接和一个交叉度数代理仍不同于原张量绑定预处理，原未发布稠密局部修正预处理未包括；差异保留于provenance。
'''
with zipfile.ZipFile(old) as src,zipfile.ZipFile(pub/(prefix+'model_pipeline.zip'),'w',zipfile.ZIP_DEFLATED) as z:
 cfg=json.loads(src.read('runtime/deployment_config.json'));cfg.update(version='TRGD-V90-saved-V35-500-v3',input_scope='Exact supplied registered500px RGBs; no registration,cropping,resizing or field reconstruction',foreground_source='Exact saved V35 masks',foreground_recomputed_by_V90=False)
 for name in src.namelist():
  if not name.startswith('runtime/') or name in ['runtime/deployment_config.json','runtime/README.md','runtime/predict_series.py','runtime/export_candidate.py']:continue
  data=src.read(name);z.writestr(name,data);inventory[name]=hashlib.sha256(data).hexdigest()
 for name,data in [('runtime/deployment_config.json',(json.dumps(cfg,indent=2)+'\n').encode()),('runtime/README.md',readme.encode())]:z.writestr(name,data);inventory[name]=hashlib.sha256(data).hexdigest()
 for file in ['02_root_background/models.py','02_root_background/infer_full_250.py','04_ConvLSTM_GAT_V90/predict_saved_v35.py']:z.write(root/file,file)
 z.write(parent,'inputs/parents.csv');z.writestr('README.md',readme);z.writestr('requirements.txt',src.read('runtime/requirements.txt'));z.writestr('runtime_checksums.json',json.dumps(inventory,indent=2)+'\n')
(pub/(prefix+'model_README.md')).write_text(readme)
for name in ['inputs','model_pipeline']:
 p=pub/(prefix+name+'.zip');assert p.stat().st_size<100*1048576
 with zipfile.ZipFile(p) as z:assert z.testzip() is None
checks={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in pub.glob(prefix+'*') if p.is_file() and not p.name.endswith('checksums.json')}
(pub/(prefix+'checksums.json')).write_text(json.dumps(checks,indent=2)+'\n');print('SAVED_V35_MODEL_PACKAGED',len(checks))
