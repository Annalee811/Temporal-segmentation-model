"""Bind current V35 common-field masks to the frozen V90/TRGD temporal transfer runtime."""
import argparse,csv,hashlib,json,os,pickle,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def immutable_copy(src,dst):
 src,dst=Path(src),Path(dst)
 if dst.exists():assert sha(src)==sha(dst);return
 try:os.link(src,dst)
 except OSError:shutil.copy2(src,dst)
def js(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)
def csvout(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def export_position(pos,source,run,dest,rows,config):
 dest.mkdir(parents=True,exist_ok=True);work=run/'work';periods=[r['period'] for r in rows];snap=pickle.load(open(work/'axis/root_axis_snapshots.pkl','rb'));tracks={};nexttrack=1;records=[]
 for i,r in enumerate(rows):
  t=r['period'];src=source/pos;im=np.array(Image.open(src/(t+'_common_RGB.png')).convert('RGB'));binary=np.array(Image.open(src/(t+'_root_mask.png')))>0;domain=np.array(Image.open(src/'common_domain.png'))>0;struct=np.array(Image.open(work/'junction/full'/(t+'_classes.png')));owner=np.load(work/'junction/ownership'/(t+'_axis_owner.npz'))['axis'];assert np.array_equal(struct>0,binary) and np.array_equal(owner>=0,binary);cl=np.where(binary,np.where(struct==2,2,1),0).astype('uint8')
  assert np.isin(cl,[0,1,2]).all() and not cl[~domain].any();assert (i!=0) or not (cl==2).any()
  overlay=im.copy();palette=np.array([[0,0,0],[55,118,185],[230,159,0]],dtype='uint8');overlay[binary]=(im[binary].astype('float32')*.25+palette[cl[binary]].astype('float32')*.75).astype('uint8');assert np.array_equal(im[~binary],overlay[~binary]);Image.fromarray(cl).save(dest/(t+'_classes.png'));Image.fromarray(overlay).save(dest/(t+'_root_temporal_overlay.png'));Image.fromarray(binary.astype('uint8')*255).save(dest/(t+'_V35_root_mask.png'));immutable_copy(src/(t+'_common_RGB.png'),dest/(t+'_common_RGB.png'))
  assigned=set()
  for rec in sorted(snap[t]['states'],key=lambda v:-v['prior_support']):
   key=(t,int(rec['axis']));prior=(rec['prior_period'],int(rec['prior_axis']))
   if rec['state']=='established' and prior in tracks and prior not in assigned:tracks[key]=tracks[prior];assigned.add(prior)
   else:tracks[key]=nexttrack;nexttrack+=1
  track=np.zeros(binary.shape,np.int32)
  lookup=np.zeros(int(owner[binary].max())+1,np.int32)
  for axis in np.unique(owner[binary]):lookup[int(axis)]=tracks[t,int(axis)]
  track[binary]=lookup[owner[binary]]
  np.savez_compressed(dest/(t+'_ownership.npz'),axis=owner,track_id=track)
  previous=periods[i-1] if i else None;meta=dict(position=pos,period=t,previous_observed_period=previous,period_gap=None if previous is None else int(t[1:])-int(previous[1:]),width=cl.shape[1],height=cl.shape[0],root_pixels=int(binary.sum()),existing_root_pixels=int((cl==1).sum()),new_root_pixels=int((cl==2).sum()),foreground_added_pixels=0,foreground_removed_pixels=0,first_observation_reference=(i==0),binary_weights_SHA256=config['weights_SHA256']['upstream/02_root_background/assets/weights/unet/best_model.pt'],V90_weights_SHA256=config['weights_SHA256']['weights/V90_joint_best.pt'],source_common_RGB_SHA256=sha(src/(t+'_common_RGB.png')),classes_SHA256=sha(dest/(t+'_classes.png')),overlay_SHA256=sha(dest/(t+'_root_temporal_overlay.png')),V35_mask_SHA256=sha(src/(t+'_root_mask.png')),actual_ConvLSTM_GAT_forward=True,axis_constraint_recomputed=True,junction_constraint_recomputed=True,temporal_weights_retrained=False,temporal_interface='existing V90 pretrained transfer adapter',classes='0=background,1=existing_root,2=new_root')
  meta.update(binary_selected_epoch=config['upstream_binary_epoch'],binary_inference_window=config.get('upstream_binary_window',256),binary_inference_stride=config.get('upstream_binary_stride',192),binary_training_epochs=config.get('upstream_binary_training_epochs',100))
  assert meta['existing_root_pixels']+meta['new_root_pixels']==meta['root_pixels'];records.append(meta);js(dest/(t+'_metadata.json'),meta)
 shutil.copy2(source/pos/'common_domain.png',dest/'common_domain.png');csvout(dest/'temporal_summary.csv',records);js(dest/'preprocessing_provenance.json',{k:v for k,v in json.load(open(work/'adapter_provenance.json')).items() if k not in ['accuracy_status','no_manual_labels']});js(dest/'completion.json',dict(position=pos,periods=periods,images=len(rows),class_codes={'0':'background','1':'existing_root','2':'new_root'},T1_reference='all visible T1 roots assigned to existing baseline',foreground_equal_latest_V35=True,method='V90 pretrained transfer interface: ConvLSTM+GAT, Axis, Junction',weights=config['weights_SHA256']));return records

def main(a):
 source=a.source.resolve();out=a.output.resolve();runtime=a.runtime.resolve();out.mkdir(parents=True,exist_ok=True);cfg=json.load(open(runtime/'deployment_config.json'));source_done=json.load(open(source/'completion.json'));assert source_done['images']==66 and source_done['positions']==18
 for rel,digest in cfg['weights_SHA256'].items():assert sha(runtime/rel)==digest,rel
 assert source_done['model_SHA256']==cfg['weights_SHA256']['upstream/02_root_background/assets/weights/unet/best_model.pt']
 rows=list(csv.DictReader(open(source/'prediction_manifest.csv',encoding='utf-8-sig')));positions=sorted({r['position'] for r in rows});assert len(positions)==18
 if a.position:positions=[a.position];assert a.position in {r['position'] for r in rows}
 allrecords=[]
 for n,pos in enumerate(positions):
  rr=sorted([r for r in rows if r['position']==pos],key=lambda r:r['period']);run=out/'runs'/pos;rgbdir=run/'RGB';priors=run/'priors'/pos;rgbdir.mkdir(parents=True,exist_ok=True);priors.mkdir(parents=True,exist_ok=True);manifest_rows=[]
  for r in rr:
   t=r['period'];rgb=source/pos/(t+'_common_RGB.png');mask=np.array(Image.open(source/pos/(t+'_root_mask.png')))>0;assert sha(rgb)==r['source_common_RGB_SHA256'] and sha(source/pos/(t+'_root_mask.png'))==r['root_mask_SHA256'];assert sha(source/pos/(t+'_root_probability.npz'))==r['probability_SHA256']
   with np.load(source/pos/(t+'_root_probability.npz')) as z:p=z['root_probability'].astype('float32')
   assert np.array_equal(p>=.5,mask);np.save(priors/(t+'_binary_probability.npy'),p);immutable_copy(rgb,rgbdir/(t+'.png'));manifest_rows.append(dict(position_id=pos,period=t,canonical_view_relative_path=t+'.png',registered_RGB_SHA256=sha(rgb),review_required=False,binary_mask_SHA256=r['root_mask_SHA256'],binary_probability_SHA256=r['probability_SHA256']))
  binding=dict(pipeline=cfg,inputs=manifest_rows,foreground_source='latest V35 common field masks',class_output='two classes; T1 existing baseline')
  p=run/'pipeline_config.json'
  if p.exists():assert json.load(open(p))==binding,'Existing run configuration changed'
  else:js(p,binding)
  manifest=run/'input_manifest.csv';csvout(manifest,manifest_rows);env=os.environ.copy();env.update(V90_INPUT_MANIFEST=str(manifest),V90_RGB_ROOT=str(rgbdir),V90_BINARY_THRESHOLD='0.5',V90_USE_VERIFIED_B3_CACHE='0',PYTHONUNBUFFERED='1');dest=out/'predictions'/pos;tic=time.monotonic()
  if not (dest/'completion.json').exists():
   commands=[('V66 temporal priors','upstream_predict.py',['--position',pos,'--output',str(run/'priors')]),('V90 graph inputs','build_graph_inputs.py',['--position',pos,'--inputs',str(run/'priors'),'--output',str(run/'work')]),('V90 ConvLSTM GAT Axis Junction','infer_candidate.py',['--work',str(run/'work')])]
   for phase,script,args in commands:
    js(out/'status.json',dict(state='running',completed_positions=n,total_positions=len(positions),position=pos,phase=phase));print('STAGE',pos,phase,flush=True)
    with (run/(script+'.log')).open('w') as log:subprocess.run([sys.executable,str(runtime/script),*args],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  exported=export_position(pos,source,run,dest,rr,cfg);allrecords.extend(exported);csvout(out/'temporal_prediction_manifest.csv',allrecords);print('POSITION_COMPLETE',pos,len(rr),'seconds',round(time.monotonic()-tic,1),flush=True)
 if not a.position:
  assert len(allrecords)==66;shutil.copy2(source/'missing_periods.csv',out/'missing_periods.csv');js(out/'completion.json',dict(state='complete',positions=18,images=66,source='common fields only',missing_periods='absent from inference; blank in four-slot figures',classes={'0':'background','1':'existing_root','2':'new_root'},T1='existing-root baseline reference',binary_model=source_done['model_checkpoint'],binary_selected_epoch=source_done['selected_epoch'],binary_window=source_done.get('binary_window',256),binary_stride=source_done.get('binary_stride',192),foreground_equal_latest_V35=True,foreground_added_pixels=0,foreground_removed_pixels=0,temporal_weights_retrained=False,temporal_interface='existing V90 pretrained transfer adapter',known_preprocessing_departures=cfg['known_departures'],weights_SHA256=cfg['weights_SHA256']));js(out/'status.json',dict(state='complete',completed_positions=18,total_positions=18))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=ROOT/'work/V35_full250_common66');p.add_argument('--output',type=Path,default=ROOT/'work/V90_full250_common66');p.add_argument('--runtime',type=Path,default=ROOT/'work/V90_full250_common66/runtime');p.add_argument('--position');main(p.parse_args())
