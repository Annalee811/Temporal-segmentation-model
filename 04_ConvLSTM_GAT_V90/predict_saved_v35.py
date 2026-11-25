"""Use already matched RGBs and saved V35 foreground; only infer temporal states."""
import argparse,csv,hashlib,json,os,shutil,subprocess,sys,time,zipfile
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def js(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)
def csvout(p,rows):
 with Path(p).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def copy_readonly(src,dst):
 src,dst=Path(src),Path(dst)
 if dst.exists():assert sha(src)==sha(dst);return
 try:os.link(src,dst)
 except OSError:shutil.copy2(src,dst)

def prepare(a,runtime):
 import torch
 sys.path.insert(0,str(ROOT/'02_root_background'))
 from models import RootUNet
 from infer_full_250 import predict_image
 source=a.output/'inputs';source.mkdir(exist_ok=True)
 rows=[r for r in csv.DictReader(open(a.parents,encoding='utf-8-sig')) if r['source']=='submitted_66']
 assert len(rows)==66 and len({r['position'] for r in rows})==18
 cfg=json.load(open(runtime/'deployment_config.json'));weights=runtime/'upstream/02_root_background/assets/weights/unet/best_model.pt'
 assert sha(weights)==cfg['weights_SHA256']['upstream/02_root_background/assets/weights/unet/best_model.pt']
 torch.set_num_threads(4);net=RootUNet().to(memory_format=torch.channels_last)
 net.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True)['state_dict']);net.eval()
 records=[]
 with zipfile.ZipFile(a.masks_zip) as z:
  for r in rows:
   pos,t=r['position'],r['period'];folder=source/pos;folder.mkdir(exist_ok=True)
   name=r['image_id'];stem=Path(name).stem;rgb_path=a.RGB_root/name
   assert sha(rgb_path)==r['RGB_sha256'];copy_readonly(rgb_path,folder/(t+'_RGB.png'))
   for suffix,target in [('_root_mask.png','_V35_mask.png'),('_root_overlay.png','_V35_overlay.png')]:
    data=z.read(stem+suffix);dst=folder/(t+target)
    if dst.exists():assert dst.read_bytes()==data
    else:dst.write_bytes(data)
   mask=np.array(Image.open(folder/(t+'_V35_mask.png')))>0;rgb=np.array(Image.open(rgb_path).convert('RGB'))
   assert mask.shape==rgb.shape[:2]==(500,500)
   pp=folder/(t+'_probability.npy')
   if pp.exists():p=np.load(pp)
   else:p=predict_image(net,rgb);np.save(pp,p)
   assert np.array_equal(p>=.5,mask),f'Soft V35 probability differs from approved mask: {name}'
   records.append(dict(position=pos,period=t,input_image=name,RGB_SHA256=r['RGB_sha256'],V35_mask_SHA256=sha(folder/(t+'_V35_mask.png')),probability_SHA256=sha(pp),root_pixels=int(mask.sum())))
   js(a.output/'status.json',dict(state='bind_saved_V35',completed_images=len(records),total_images=66));print('BOUND_SAVED_V35',len(records),name,flush=True)
 csvout(source/'manifest.csv',records)
 js(source/'completion.json',dict(images=66,positions=18,source_zip=str(a.masks_zip),source_zip_SHA256=sha(a.masks_zip),foreground='Exact saved model masks, not submitted annotation masks',re_registered=False,re_cropped=False,resized=False,soft_probability_check='Fresh same500px V35 inference threshold matches each saved mask exactly'))
 return source,records,cfg

def main(a):
 a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True);runtime=a.runtime.resolve()
 source,rows,cfg=prepare(a,runtime);records=[];positions=sorted({r['position'] for r in rows})
 for i,pos in enumerate(positions):
  rr=sorted([r for r in rows if r['position']==pos],key=lambda r:r['period']);run=a.output/'runs'/pos;rgbdir=run/'RGB';prior=run/'priors'/pos
  rgbdir.mkdir(parents=True,exist_ok=True);prior.mkdir(parents=True,exist_ok=True);manifest=[]
  for r in rr:
   t=r['period'];copy_readonly(source/pos/(t+'_RGB.png'),rgbdir/(t+'.png'));copy_readonly(source/pos/(t+'_probability.npy'),prior/(t+'_binary_probability.npy'))
   manifest.append(dict(position_id=pos,period=t,canonical_view_relative_path=t+'.png',registered_RGB_SHA256=r['RGB_SHA256'],review_required=False))
  csvout(run/'input_manifest.csv',manifest);binding=dict(pipeline=cfg,inputs=rr,foreground_policy='Frozen saved V35 masks on unchanged supplied RGBs')
  if (run/'pipeline_config.json').exists():assert json.load(open(run/'pipeline_config.json'))==binding
  else:js(run/'pipeline_config.json',binding)
  env=os.environ.copy();env.update(V90_INPUT_MANIFEST=str(run/'input_manifest.csv'),V90_RGB_ROOT=str(rgbdir),V90_USE_VERIFIED_B3_CACHE='0',V90_BINARY_THRESHOLD='0.5',PYTHONUNBUFFERED='1');env.pop('V90_COMMON_DOMAIN',None)
  dest=a.output/'predictions'/pos;dest.mkdir(parents=True,exist_ok=True)
  if not (dest/'completion.json').exists():
   for phase,script,args in [('temporal_priors','upstream_predict.py',['--position',pos,'--output',str(run/'priors')]),('graph','build_graph_inputs.py',['--position',pos,'--inputs',str(run/'priors'),'--output',str(run/'work')]),('ConvLSTM_GAT_Axis_Junction','infer_candidate.py',['--work',str(run/'work')])]:
    js(a.output/'status.json',dict(state='temporal_prediction',completed_positions=i,total_positions=18,position=pos,phase=phase))
    with (run/(script+'.log')).open('w') as log:subprocess.run([sys.executable,str(runtime/script),*args],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  per=[]
  for j,r in enumerate(rr):
   t=r['period'];rgb=np.array(Image.open(source/pos/(t+'_RGB.png')).convert('RGB'));mask=np.array(Image.open(source/pos/(t+'_V35_mask.png')))>0
   structural=np.array(Image.open(run/'work/junction/full'/(t+'_classes.png')));assert np.array_equal(structural>0,mask)
   cl=np.where(mask,np.where(structural==2,2,1),0).astype('uint8');assert np.isin(cl,[0,1,2]).all() and np.array_equal(cl>0,mask)
   assert j!=0 or not (cl==2).any()
   overlay=rgb.copy();palette=np.array([[0,0,0],[55,118,185],[230,159,0]],np.uint8);overlay[mask]=(rgb[mask]*.25+palette[cl[mask]]*.75).astype('uint8');assert np.array_equal(overlay[~mask],rgb[~mask])
   Image.fromarray(cl).save(dest/(t+'_V90_classes.png'));Image.fromarray(overlay).save(dest/(t+'_V90_overlay.png'))
   for suffix in ['_RGB.png','_V35_mask.png','_V35_overlay.png']:copy_readonly(source/pos/(t+suffix),dest/(t+suffix))
   assert sha(dest/(t+'_V35_mask.png'))==r['V35_mask_SHA256']
   previous=rr[j-1]['period'] if j else None
   meta=dict(**r,width=500,height=500,previous_observed_period=previous,period_gap=None if previous is None else int(t[1:])-int(previous[1:]),existing_root_pixels=int((cl==1).sum()),new_root_pixels=int((cl==2).sum()),foreground_difference_pixels=0,RGB_resized=False,RGB_re_registered=False,RGB_re_cropped=False,foreground_source='Exact user-selected V35 prediction ZIP',binary_selected_epoch=91,V90_temporal_weights_retrained=False,actual_ConvLSTM_GAT_Axis_Junction=True,weights_SHA256=cfg['weights_SHA256'])
   js(dest/(t+'_metadata.json'),meta);per.append(meta);records.append(meta)
  csvout(dest/'temporal_summary.csv',per)
  provenance=json.load(open(run/'work/adapter_provenance.json'));js(dest/'preprocessing_provenance.json',{k:v for k,v in provenance.items() if k not in ['accuracy_status','no_manual_labels']})
  js(dest/'completion.json',dict(position=pos,periods=[r['period'] for r in rr],images=len(rr),foreground_equals_saved_V35=True))
  csvout(a.output/'prediction_manifest.csv',records);print('POSITION_COMPLETE',i+1,pos,len(per),flush=True)
 assert len(records)==66
 missing=[dict(position=pos,period=t) for pos in positions for t in ['T1','T2','T3','T4'] if not any(r['position']==pos and r['period']==t for r in rows)];assert len(missing)==6;csvout(a.output/'missing_periods.csv',missing)
 js(a.output/'completion.json',dict(state='complete',positions=18,images=66,width=500,height=500,source_V35_zip_SHA256=sha(a.masks_zip),foreground_difference_pixels=0,provided_RGB_preserved=True,resized=False,re_registered=False,re_cropped=False,classes={'0':'background','1':'existing_root','2':'new_root'},missing_periods=6,weights_SHA256=cfg['weights_SHA256'],temporal_interface='Existing V90 pretrained transfer interface',known_departures=cfg['known_departures']))
 js(a.output/'status.json',dict(state='complete',completed_positions=18,images=66));print('ALL66_SAVED_V35_TEMPORAL_COMPLETE',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--masks-zip',type=Path,default=Path('/workspace/shared/downloads/V35_full250_all66_supplemental_predictions.zip'))
 p.add_argument('--parents',type=Path,default=ROOT/'work/V35_full250_all66/data/parents.csv');p.add_argument('--RGB-root',type=Path,default=ROOT/'work/V35_310_100epochs/data/RGB')
 p.add_argument('--runtime',type=Path,default=ROOT/'work/V90_full250_common66/runtime');p.add_argument('--output',type=Path,default=ROOT/'work/V90_saved_V35_66');main(p.parse_args())
