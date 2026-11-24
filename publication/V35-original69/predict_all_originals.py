import argparse,csv,hashlib,json,time,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path('/workspace/Temporal-segmentation-model');sys.path.insert(0,str(ROOT/'02_root_background'));from models import RootUNet

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def js(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def csvout(p,rows):
 q=Path(p).with_suffix('.tmp')
 with q.open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 q.replace(p)
def main(a):
 out=a.output;out.mkdir(parents=True,exist_ok=True);sources=json.load(open('/tmp/LI-raw69/raw_manifest.json'));assert len(sources)==69;training=json.load(open(ROOT/'work/V35_310_100epochs/completion.json'));weights=ROOT/'work/V35_310_100epochs/weights/V35_310_best.pt';model_hash=sha(weights);assert model_hash==training['best_weights_SHA256']
 torch.set_num_threads(a.threads);model=RootUNet().to(memory_format=torch.channels_last);model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True)['state_dict']);model.eval();records=[];started=time.monotonic();win=np.maximum(np.hanning(256)[:,None]*np.hanning(256)[None,:],.02).astype('float32')
 @torch.inference_mode()
 def predict(im,base_status):
  h,w=im.shape[:2];ph=max(256,h);pw=max(256,w);im=np.pad(im,((0,ph-h),(0,pw-w),(0,0)),mode='edge');xs=sorted(set(list(range(0,pw-255,192))+[pw-256]));ys=sorted(set(list(range(0,ph-255,192))+[ph-256]));coords=[(x,y) for y in ys for x in xs];sm=np.zeros((ph,pw),np.float32);wg=np.zeros_like(sm)
  for i in range(0,len(coords),2):
   cc=coords[i:i+2];x=torch.from_numpy(np.stack([im[y:y+256,x:x+256] for x,y in cc]).transpose(0,3,1,2).copy()).float().to(memory_format=torch.channels_last)/255;pp=model(x).sigmoid()[:,0].numpy()
   for (x,y),p in zip(cc,pp):sm[y:y+256,x:x+256]+=p*win;wg[y:y+256,x:x+256]+=win
   if i%40==0:js(out/'status.json',dict(**base_status,processed_tiles=min(i+2,len(coords)),total_tiles=len(coords)))
  return (sm/wg)[:h,:w]
 for i,r in enumerate(sources):
  pos=r['position'];t='T'+str(r['period']);folder=out/pos;folder.mkdir(exist_ok=True);meta_path=folder/(t+'_metadata.json');path=Path(r['path']);assert sha(path)==r['sha256'];mask_path=folder/(t+'_root_mask.png');overlay_path=folder/(t+'_root_overlay.png');prob_path=folder/(t+'_root_probability.npz')
  if meta_path.exists():
   d=json.load(open(meta_path));assert d['model_SHA256']==model_hash and d['source_file_SHA256']==r['sha256'];assert all(sha(p)==d[k] for p,k in [(mask_path,'mask_SHA256'),(overlay_path,'overlay_SHA256'),(prob_path,'probability_file_SHA256')]);records.append(d);print('REUSE',i+1,pos,t,flush=True);continue
  im=np.array(Image.open(path).convert('RGB'));h,w=im.shape[:2];assert (w,h)==(r['width'],r['height']);status=dict(state='predicting',completed_images=i,total_images=69,position=pos,period=t,width=w,height=h,image_scope='complete native original image');js(out/'status.json',status);tic=time.monotonic();p=predict(im,status);assert p.shape==(h,w) and p.dtype==np.float32 and np.isfinite(p).all();mask=p>=.5;Image.fromarray(mask.astype('uint8')*255).save(mask_path);np.savez_compressed(prob_path,root_probability=p)
  overlay=im.copy();overlay[mask]=(im[mask].astype('float32')*.25+np.array([255,193,7],dtype='float32')*.75).astype('uint8');assert np.array_equal(overlay[~mask],im[~mask]);Image.fromarray(overlay).save(overlay_path)
  d=dict(position=pos,material=pos.split('-')[0],tube=pos.rsplit('-',1)[0],period=t,width=w,height=h,pixels=w*h,predicted_root_pixels=int(mask.sum()),root_area_fraction=float(mask.mean()),source_archive=r['source_archive'],source_archive_entry=r['archive_entry'],source_file_name=path.name,source_file_SHA256=r['sha256'],decoded_rgb_SHA256=hashlib.sha256(im.tobytes()).hexdigest(),model_checkpoint='V35_310_best.pt',model_SHA256=model_hash,training_epochs=100,selected_epoch=training['selected_epoch'],threshold=.5,classes='0=background,255=root',image_scope='complete native original input, not common-field RGB',registration_applied=False,resizing_applied=False,postprocessing_applied=False,mask_SHA256=sha(mask_path),overlay_SHA256=sha(overlay_path),probability_file_SHA256=sha(prob_path),probability_array_SHA256=hashlib.sha256(p.tobytes()).hexdigest(),prediction_seconds=time.monotonic()-tic)
  js(meta_path,d);records.append(d);csvout(out/'prediction_manifest.csv',records);print('PREDICTED',i+1,'/69',pos,t,w,h,'root_pixels',d['predicted_root_pixels'],'seconds',round(d['prediction_seconds'],1),flush=True)
 positions=sorted({r['position'] for r in records});missing=[dict(position=pos,period=t,reason='No original source observation') for pos in positions for t in ['T1','T2','T3','T4'] if not any(r['position']==pos and r['period']==t for r in records)];csvout(out/'missing_observations.csv',missing);csvout(out/'prediction_manifest.csv',records);assert len(records)==69 and len(positions)==19
 js(out/'completion.json',dict(state='complete',images=69,positions=19,physical_tubes=len({r['tube'] for r in records}),source_manifest_SHA256=sha('/tmp/LI-raw69/raw_manifest.json'),source_repository='Annalee811/LI',source_commit='cd161aeebe942f515ee98c88881991ad02784631',image_scope='all native original input images listed by source_manifest.json',model_checkpoint='V35_310_best.pt',model_SHA256=model_hash,training_epochs=100,selected_epoch=training['selected_epoch'],threshold=.5,classes='root/background only',native_resolution_preserved=True,missing_observations=missing,independent_accuracy_not_evaluated=True,seconds=time.monotonic()-started));js(out/'status.json',dict(state='complete',completed_images=69,total_images=69));print('ALL69_COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--threads',type=int,default=4);main(p.parse_args())
