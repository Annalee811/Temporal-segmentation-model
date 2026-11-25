"""Verify cached inference on the exact matched native RGBs and fixed V35 masks."""
import argparse,csv,hashlib,json,os,shutil
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def link(src,dst):
 if dst.exists():assert sha(src)==sha(dst);return
 try:os.link(src,dst)
 except OSError:shutil.copy2(src,dst)
def js(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def csvout(p,rows):
 with p.open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main(a):
 a.output.mkdir(parents=True,exist_ok=True)
 source_done=json.load(open(a.binary/'completion.json'));cache_done=json.load(open(a.temporal/'completion.json'))
 assert source_done['images']==cache_done['images']==66 and source_done['positions']==cache_done['positions']==18
 assert source_done['model_SHA256']==cache_done['weights_SHA256']['upstream/02_root_background/assets/weights/unet/best_model.pt']
 assert source_done['selected_epoch']==91 and cache_done['binary_selected_epoch']==91
 rows=list(csv.DictReader(open(a.binary/'prediction_manifest.csv',encoding='utf-8-sig')));records=[]
 for i,r in enumerate(rows):
  pos,t=r['position'],r['period'];b=a.binary/pos;c=a.temporal/'predictions'/pos;d=a.output/'predictions'/pos;d.mkdir(parents=True,exist_ok=True)
  old=json.load(open(c/(t+'_metadata.json')))
  assert old['source_common_RGB_SHA256']==r['source_common_RGB_SHA256']==sha(b/(t+'_common_RGB.png'))==sha(c/(t+'_common_RGB.png'))
  assert old['binary_weights_SHA256']==source_done['model_SHA256'] and old['V90_weights_SHA256']==cache_done['weights_SHA256']['weights/V90_joint_best.pt']
  assert old['V35_mask_SHA256']==r['root_mask_SHA256']==sha(b/(t+'_root_mask.png'))
  assert sha(c/(t+'_classes.png'))==old['classes_SHA256'] and sha(c/(t+'_root_temporal_overlay.png'))==old['overlay_SHA256']
  rgb=np.array(Image.open(b/(t+'_common_RGB.png')).convert('RGB'));mask=np.array(Image.open(b/(t+'_root_mask.png')))>0;cl=np.array(Image.open(c/(t+'_classes.png')))
  assert cl.shape==mask.shape==rgb.shape[:2]==(int(r['height']),int(r['width']))
  assert np.isin(cl,[0,1,2]).all() and np.array_equal(cl>0,mask)
  assert t!='T1' or not (cl==2).any()
  for path in [b/(t+'_root_overlay.png'),c/(t+'_root_temporal_overlay.png')]:
   ov=np.array(Image.open(path).convert('RGB'));assert np.array_equal(rgb[~mask],ov[~mask])
  for src,name in [(b/(t+'_common_RGB.png'),t+'_RGB.png'),(b/(t+'_root_mask.png'),t+'_V35_mask.png'),(b/(t+'_root_overlay.png'),t+'_V35_overlay.png'),(c/(t+'_classes.png'),t+'_V90_classes.png'),(c/(t+'_root_temporal_overlay.png'),t+'_V90_overlay.png')]:link(src,d/name)
  meta=dict(position=pos,period=t,width=int(r['width']),height=int(r['height']),RGB_SHA256=r['source_common_RGB_SHA256'],V35_mask_SHA256=r['root_mask_SHA256'],V90_classes_SHA256=old['classes_SHA256'],binary_weights_SHA256=old['binary_weights_SHA256'],V90_weights_SHA256=old['V90_weights_SHA256'],root_pixels=int(mask.sum()),existing_root_pixels=int((cl==1).sum()),new_root_pixels=int((cl==2).sum()),foreground_difference_pixels=0,previous_observed_period=old['previous_observed_period'],period_gap=old['period_gap'],RGB_resized=False,RGB_re_registered=False,RGB_re_cropped=False,execution='Reused verified completed inference on exactly the same inputs and model weights',source_binary_metadata=str(b/(t+'_metadata.json')),source_temporal_metadata=str(c/(t+'_metadata.json')))
  js(d/(t+'_metadata.json'),meta);records.append(meta);csvout(a.output/'prediction_manifest.csv',records);js(a.output/'status.json',dict(state='verify_native_inputs',completed_images=i+1,total_images=66));print('VERIFIED_NATIVE',i+1,pos,t,r['width'],r['height'],flush=True)
 positions=sorted({r['position'] for r in records});missing=[]
 for pos in positions:
  rr=[r for r in records if r['position']==pos];assert len({(r['width'],r['height']) for r in rr})==1
  csvout(a.output/'predictions'/pos/'temporal_summary.csv',rr)
  for t in ['T1','T2','T3','T4']:
   if not any(r['period']==t for r in rr):missing.append(dict(position=pos,period=t))
 assert len(records)==66 and len(positions)==18 and len(missing)==6
 csvout(a.output/'missing_periods.csv',missing)
 js(a.output/'completion.json',dict(state='complete',positions=18,images=66,missing_periods=6,native_input_sizes_preserved=True,RGB_re_registered=False,RGB_re_cropped=False,RGB_resized=False,foreground_difference_pixels=0,foreground_provider='V35 prediction on the same complete matched input RGB',execution='Verified cache reuse; no new neural forward was required for identical inputs and weights',weights_SHA256=cache_done['weights_SHA256'],classes={'0':'background','1':'existing_root','2':'new_root'},temporal_interface=cache_done['temporal_interface'],known_preprocessing_departures=cache_done['known_preprocessing_departures']))
 js(a.output/'status.json',dict(state='complete',completed_images=66,positions=18));print('ALL66_MATCHED_NATIVE_VERIFIED',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,default=ROOT/'work/V35_full250_common66');p.add_argument('--temporal',type=Path,default=ROOT/'work/V90_full250_common66');p.add_argument('--output',type=Path,default=ROOT/'work/V90_all_matched_native');main(p.parse_args())
