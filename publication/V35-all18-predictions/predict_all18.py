import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import sys
ROOT=Path('/workspace/Temporal-segmentation-model');sys.path.insert(0,str(ROOT/'02_root_background'))
from models import RootUNet
from infer_cross_tube import predict_image

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def csvout(p,rows):
 with Path(p).open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main(a):
 out=a.output;out.mkdir(parents=True,exist_ok=True);ck=ROOT/'work/V35_310_100epochs/weights/V35_310_best.pt';completion=json.load(open(ROOT/'work/V35_310_100epochs/completion.json'));assert completion['epochs']==100 and sha(ck)==completion['best_weights_SHA256']
 torch.set_num_threads(a.threads);model=RootUNet().to(memory_format=torch.channels_last);model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True)['state_dict']);model.eval();source=Path('/tmp/all18-atlas/RGB');files=sorted(source.glob('*/*.png'));assert len(files)==66
 # Reuse verified complete-field source hashes from the submitted annotation provenance.
 annotation=json.load(open(ROOT/'work/V35_310_100epochs/uploaded_annotation.json'));lookup={(x['annotation_provenance']['position'],x['annotation_provenance']['period']):x['annotation_provenance'] for x in annotation['images']}
 records=[];missing=[];start=time.monotonic()
 for i,path in enumerate(files):
  pos=path.parent.name;t=path.stem;prov=lookup[pos,t];source_hash=sha(path);assert source_hash==prov['source_rgb_sha256'];folder=out/pos;folder.mkdir(exist_ok=True);meta_path=folder/(t+'_metadata.json');prob_path=folder/(t+'_root_probability.npz');mask_path=folder/(t+'_root_mask.png');overlay_path=folder/(t+'_root_overlay.png')
  if meta_path.exists():
   d=json.load(open(meta_path));assert d['model_SHA256']==sha(ck) and d['source_rgb_SHA256']==source_hash
   assert all(sha(p)==d[k] for p,k in [(mask_path,'mask_SHA256'),(prob_path,'probability_file_SHA256'),(overlay_path,'overlay_SHA256')]);records.append(d);print('REUSE',pos,t,flush=True);continue
  im=np.array(Image.open(path).convert('RGB'));h,w=im.shape[:2];js(out/'status.json',dict(state='predicting',completed_images=i,total_images=66,position=pos,period=t,width=w,height=h))
  tic=time.monotonic();p=predict_image(model,im,batch_size=2);assert p.shape==(h,w) and p.dtype==np.float32 and np.isfinite(p).all() and p.min()>=0 and p.max()<=1;mask=p>=.5;native=mask.astype('uint8')*255;Image.fromarray(native).save(mask_path);np.savez_compressed(prob_path,root_probability=p)
  overlay=im.copy();overlay[mask]=(im[mask].astype('float32')*.25+np.array([255,193,7],dtype='float32')*.75).astype('uint8');Image.fromarray(overlay).save(overlay_path)
  # Verify masks directly against stored full-float probabilities and preserve all dimensions.
  with np.load(prob_path) as z:assert np.array_equal(z['root_probability'],p)
  assert np.array_equal(np.array(Image.open(mask_path)),native);assert Image.open(overlay_path).size==(w,h)
  d=dict(position=pos,tube=prov['tube'],material=prov['material'],period=t,width=w,height=h,pixels=w*h,predicted_root_pixels=int(mask.sum()),root_area_fraction=float(mask.mean()),source_rgb_SHA256=source_hash,model_SHA256=sha(ck),model_checkpoint='V35_310_best.pt',selected_epoch=completion['selected_epoch'],training_epochs=100,threshold=.5,classes='0=background,255=root',mask_SHA256=sha(mask_path),probability_file_SHA256=sha(prob_path),probability_array_SHA256=hashlib.sha256(p.tobytes()).hexdigest(),overlay_SHA256=sha(overlay_path),prediction_seconds=time.monotonic()-tic)
  js(meta_path,d);records.append(d);csvout(out/'prediction_manifest.csv',records);print('PREDICTED',i+1,'/66',pos,t,'root_pixels',d['predicted_root_pixels'],'seconds',round(d['prediction_seconds'],2),flush=True)
 for pos in sorted({p.parent.name for p in files}):
  for t in ['T1','T2','T3','T4']:
   if (pos,t) not in lookup:missing.append(dict(position=pos,period=t,reason='No source observation supplied'))
 csvout(out/'missing_observations.csv',missing);csvout(out/'prediction_manifest.csv',records);assert len(records)==66
 summary=[]
 for pos in sorted({r['position'] for r in records}):
  rr=[r for r in records if r['position']==pos];summary.append(dict(position=pos,tube=rr[0]['tube'],material=rr[0]['material'],observed_periods=','.join(r['period'] for r in rr),observed_images=len(rr),mean_root_area_fraction=float(np.mean([r['root_area_fraction'] for r in rr])),total_root_pixels=sum(r['predicted_root_pixels'] for r in rr),total_pixels=sum(r['pixels'] for r in rr)))
 csvout(out/'position_summary.csv',summary);js(out/'completion.json',dict(state='complete',images=66,positions=18,missing_observations=missing,classes='root/background only',threshold=.5,native_resolution=True,model_checkpoint='V35_310_best.pt',selected_epoch=completion['selected_epoch'],training_epochs=100,model_SHA256=sha(ck),no_morphology_or_manual_postprocessing=True,seconds=time.monotonic()-start));js(out/'status.json',dict(state='complete',completed_images=66,total_images=66));print('ALL_66_COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--threads',type=int,default=4);main(p.parse_args())
