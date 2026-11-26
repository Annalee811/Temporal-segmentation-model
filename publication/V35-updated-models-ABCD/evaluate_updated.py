"""Measure latest V35 and preserved baselines on the same native parent images."""
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from sklearn.metrics import roc_curve,roc_auc_score,r2_score,mean_squared_error,mean_absolute_error
from skimage.morphology import skeletonize
ROOT=Path('/workspace/Temporal-segmentation-model');HERE=Path(__file__).resolve().parent;RUN=ROOT/'work/V35_full250_all66';SOURCE=ROOT/'work/V35_310_100epochs/data'
sys.path.insert(0,str(ROOT/'02_root_background'))
from models import RootUNet,RootSegFormer
from infer_full_250 import predict_image
import common as c
from deeplab_npz_forward import DeepLabNPZ

def length(mask):return int(skeletonize(np.ascontiguousarray(mask,dtype=bool)).sum())
def savejson(name,obj):(HERE/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--models',nargs='+',default=['unet']);a=ap.parse_args();torch.set_num_threads(4)
 rows=c.csvread(RUN/'data/parents.csv')
 ref={}
 for r in rows:
  gt=np.array(Image.open(SOURCE/'mask'/r['image_id']));assert c.sha(SOURCE/'mask'/r['image_id'])==r['mask_sha256']
  ref[r['image_id']]=(gt,length(gt==255))
 for model in a.models:
  if (HERE/(model+'_complete.json')).exists():print('CACHED_COMPLETE',model,flush=True);continue
  if model=='unet':
   ck=RUN/'weights/V35_full250_best.pt';net=RootUNet().to(memory_format=torch.channels_last);net.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True)['state_dict'])
  elif model=='segformer':
   ck=ROOT/'02_root_background/assets/weights/segformer/best_model.pt';net=RootSegFormer();net.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True)['state_dict'])
  else:
   eq=c.read(HERE/'deeplab_equivalence.json');assert eq['max_abs']<1e-4 and eq['total_threshold_disagreement']==0
   ck=ROOT/'02_root_background/assets/weights/deeplab/best_model.npz';net=DeepLabNPZ(ck.parent)
  net.eval();records=[];yy=[];pp=[];counts={'train':np.zeros(4,np.int64),'val':np.zeros(4,np.int64)};t=time.time()
  for i,r in enumerate(rows,1):
   gt,gl=ref[r['image_id']];id=r['image_id'];cached=RUN/'validation_predictions/selected'/(id+'_probability.npy')
   if model=='unet' and r['split']=='val':prob=np.load(cached)
   elif model=='unet' and r['source']=='submitted_66':
    mask=np.array(Image.open(RUN/'supplemental_predictions'/(Path(id).stem+'_root_mask.png')))>0;prob=None
   else:
    im=np.array(Image.open(SOURCE/'RGB'/id).convert('RGB'));assert c.sha(SOURCE/'RGB'/id)==r['RGB_sha256'];prob=predict_image(net,im,8)
   if prob is not None:mask=prob>=.5
   assert mask.shape==gt.shape
   cc=c.counts(gt,mask);counts[r['split']]+=cc
   records.append(dict(model=model,image_id=id,split=r['split'],source=r['source'],position=r['position'],period=r['period'],width=gt.shape[1],height=gt.shape[0],reference_length_px=gl,predicted_length_px=length(mask),**c.metrics(cc)))
   if r['split']=='val':
    assert prob is not None;use=gt!=128;yy.append((gt[use]==255).astype('uint8'));pp.append(prob[use])
   if i%20==0:print(model,i,len(rows),'seconds',round(time.time()-t,1),flush=True)
  c.csvout(HERE/(model+'_parent_metrics.csv'),records)
  y=np.concatenate(yy);p=np.concatenate(pp);fpr,tpr,thr=roc_curve(y,p,drop_intermediate=True);auc=float(roc_auc_score(y,p));np.savez_compressed(HERE/(model+'_roc.npz'),fpr=fpr,tpr=tpr,thresholds=thr)
  result=dict(model=model,checkpoint=str(ck.relative_to(ROOT)),checkpoint_SHA256=c.sha(ck),input_window=250,stride=188,threshold=.5,parent_count=310,training_parents=262,monitoring_parents=48,independent_test=False,monitoring_auc=auc,monitoring_roc_pixels=len(y),monitoring_root_pixels=int(y.sum()),counts={k:c.metrics(v) for k,v in counts.items()},length_definition='Number of foreground pixels in a one-pixel-wide Zhang skeleton, without pruning or resampling',regression={})
  for split in ['train','val']:
   rs=[r for r in records if r['split']==split];x=np.array([r['reference_length_px'] for r in rs]);z=np.array([r['predicted_length_px'] for r in rs]);coef=np.polyfit(x,z,1);fit=np.polyval(coef,x)
   result['regression'][split]=dict(n=len(rs),R2=float(r2_score(x,z)),RMSE=float(np.sqrt(mean_squared_error(x,z))),MAE=float(mean_absolute_error(x,z)),OLS_R2=float(r2_score(z,fit)),slope=float(coef[0]),intercept=float(coef[1]))
  if model=='unet':
   expected=c.read(RUN/'selection.json')['overall'];assert all(result['counts']['val'][k]==expected[k] for k in ['TP','FP','FN','TN'])
  savejson(model+'_complete.json',result);print('MODEL_COMPLETE',model,'AUC',auc,'counts',result['counts']['val'],flush=True)
if __name__=='__main__':main()
