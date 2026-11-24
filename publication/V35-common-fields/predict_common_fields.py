import csv,hashlib,io,json,time,zipfile,sys,shutil
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path('/workspace/Temporal-segmentation-model');sys.path.insert(0,str(ROOT/'02_root_background'));from models import RootUNet
from infer_cross_tube import predict_image
out=ROOT/'work/V35_four_period_common';out.mkdir(exist_ok=True);archive=zipfile.ZipFile('/tmp/LI-raw69/all_series_original_and_stitched_RGB_v2.zip');summary=list(csv.DictReader(open('/tmp/LI-all-all_series_position_summary.csv',encoding='utf-8-sig')));positions=sorted(r['position'] for r in summary if int(r['independent_RGB_images'])>0);assert len(positions)==18
sha=lambda b:hashlib.sha256(b).hexdigest()
def js(p,d):
 p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def csvout(p,rr):
 with Path(p).open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
weights=ROOT/'work/V35_310_100epochs/weights/V35_310_best.pt';model_hash=sha(weights.read_bytes());assert model_hash==json.load(open(ROOT/'work/V35_310_100epochs/completion.json'))['best_weights_SHA256'];torch.set_num_threads(4);model=RootUNet().to(memory_format=torch.channels_last);model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True)['state_dict']);model.eval();records=[];start=time.monotonic()
for pos in positions:
 folder=out/pos;folder.mkdir(exist_ok=True);prefix='all_series_localization_v2/'+pos+'/';domain_data=archive.read(prefix+'masks/stitched_geometric_overlap_mask.png');domain=np.array(Image.open(io.BytesIO(domain_data)))>0;Image.fromarray(domain.astype('uint8')*255).save(folder/'common_domain.png');geometry=archive.read(prefix+'geometry_metadata.json');(folder/'geometry_metadata.json').write_bytes(geometry);geom=json.loads(geometry);assert geom['regions'] and set(geom['periods'])<=set([1,2,3,4]);expected_shape=None
 for t in ['T'+str(n) for n in sorted(geom['periods'])]:
  data=archive.read(prefix+'RGB/'+t+'_independent_stitched_RGB.png');source_hash=sha(data);im=np.array(Image.open(io.BytesIO(data)).convert('RGB'));h,w=im.shape[:2];assert domain.shape==(h,w)
  if expected_shape is None:expected_shape=(h,w)
  assert expected_shape==(h,w);(folder/(t+'_common_RGB.png')).write_bytes(data)
  existing=folder/(t+'_metadata.json')
  if existing.exists():
   saved=json.load(open(existing))
   if saved['source_common_RGB_SHA256']==source_hash and saved['model_SHA256']==model_hash and saved['domain_mask_SHA256']==sha((folder/'common_domain.png').read_bytes()):
    records.append(saved);print('REUSE_COMMON',len(records),pos,t,flush=True);continue
  js(out/'status.json',dict(state='predicting_common_fields',completed_images=len(records),total_images=66,position=pos,period=t,width=w,height=h))
  cached=ROOT/'work/V35_310_all18_predictions'/pos;meta_path=cached/(t+'_metadata.json');cache_used=False
  if meta_path.exists():
   meta=json.load(open(meta_path))
   if meta['source_rgb_SHA256']==source_hash and meta['model_SHA256']==model_hash:
    with np.load(cached/(t+'_root_probability.npz')) as z:p=z['root_probability']
    cache_used=True
   else:p=predict_image(model,im)
  else:p=predict_image(model,im)
  assert p.shape==(h,w) and p.dtype==np.float32 and np.isfinite(p).all();root=(p>=.5)&domain;mask=root.astype('uint8')*255;Image.fromarray(mask).save(folder/(t+'_root_mask.png'));common_p=np.where(domain,p,0).astype('float32');np.savez_compressed(folder/(t+'_root_probability.npz'),root_probability=common_p,common_domain=domain.astype('uint8'))
  overlay=im.copy();overlay[root]=(im[root].astype('float32')*.25+np.array([255,193,7],dtype='float32')*.75).astype('uint8');Image.fromarray(overlay).save(folder/(t+'_root_overlay.png'));assert not root[~domain].any() and np.array_equal(overlay[~root],im[~root])
  d=dict(position=pos,material=pos.split('-')[0],period=t,width=w,height=h,shared_domain_pixels=int(domain.sum()),predicted_root_pixels=int(root.sum()),root_fraction_of_common_domain=float(root.sum()/domain.sum()),source_common_RGB_SHA256=source_hash,model_SHA256=model_hash,training_epochs=100,selected_epoch=32,threshold=.5,four_periods_present=(sorted(geom['periods'])==[1,2,3,4]),domain='existing geometric common field across observed periods at native T1 reference scale',noncommon_original_image_delivered=False,additional_RGB_resizing=False,additional_registration=False,geometric_mask_applied=True,domain_mask_SHA256=sha((folder/'common_domain.png').read_bytes()),root_mask_SHA256=sha((folder/(t+'_root_mask.png')).read_bytes()),overlay_SHA256=sha((folder/(t+'_root_overlay.png')).read_bytes()),probability_SHA256=sha((folder/(t+'_root_probability.npz')).read_bytes()),cached_identical_input_and_model_prediction=cache_used)
  js(folder/(t+'_metadata.json'),d);records.append(d);csvout(out/'prediction_manifest.csv',records);print('COMMON_PREDICTED',len(records),pos,t,'cache',cache_used,flush=True)
excluded=[dict(position=r['position'],observed_periods=r['observed_periods'],reason='No established common observation field') for r in summary if r['position'] not in positions];csvout(out/'excluded_positions.csv',excluded);csvout(out/'prediction_manifest.csv',records);assert len(records)==66
js(out/'completion.json',dict(state='complete',positions=18,images=66,period_slots=['T1','T2','T3','T4'],scope='common fields across observed periods; missing periods remain blank',model_checkpoint='V35_310_best.pt',model_SHA256=model_hash,training_epochs=100,selected_epoch=32,threshold=.5,original_noncommon_field_excluded=True,separators_excluded_from_root_masks=True,common_RGB_existing_pixels_preserved=True,RGB_resampled_by_this_export=False,excluded_positions=excluded,seconds=time.monotonic()-start));js(out/'status.json',dict(state='complete',completed_images=66,total_images=66));print('ALL_OBSERVED_COMMON_COMPLETE',flush=True)
