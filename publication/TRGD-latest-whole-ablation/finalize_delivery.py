from pathlib import Path
import shutil,json,hashlib,zipfile
W=Path(__file__).resolve().parent;P=Path('/workspace/shared/downloads');K='TRGD_latest_whole_pipeline_ablation'
shutil.copy2(W/'preview.png',P/(K+'_preview.png'))
files=list(P.glob(K+'*'))+list(P.glob('V35_TRGD_*ablation*'))
with zipfile.ZipFile(P/(K+'_bundle.zip'),'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f in sorted(set(files)):
  if not f.name.endswith(('_bundle.zip','_checksums.json')):z.write(f,f.name)
 for name in ['evaluate_whole_pipeline.py','render_ablation.py','finalize_delivery.py']:z.write(W/name,'code/'+name)
files=list(P.glob(K+'*'))+list(P.glob('V35_TRGD_*ablation*'))
checks={f.name:dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in set(files) if not f.name.endswith('_checksums.json')};(P/(K+'_checksums.json')).write_text(json.dumps(checks,indent=2)+'\n');print('ARTIFACT_COUNT',len(checks))
