"""Inference-only TF1.12 slim DeepLab graph translation; original NPZ is immutable.
Validated numerically against archived native256 TF probabilities before use.
No parameter conversion is saved and no optimization is performed.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

class DeepLabNPZ:
 def __init__(self,folder):
  names=json.loads((Path(folder)/'weight_names.json').read_text())
  with np.load(Path(folder)/'best_model.npz') as z:
   self.v={name.removesuffix(':0'):torch.from_numpy(z['v%04d'%i].copy()) for i,name in enumerate(names)}
  self.used=set()
  for k in self.v:
   if k.endswith('/weights'):self.v[k]=self.v[k].permute(3,2,0,1).contiguous()
 def get(self,k):self.used.add(k);return self.v[k]
 def bn(self,x,n):
  return F.batch_norm(x,self.get(n+'/moving_mean'),self.get(n+'/moving_variance'),self.get(n+'/gamma'),self.get(n+'/beta'),training=False,eps=1e-5)
 def conv(self,x,n,s=1,d=1,norm=True,act=True):
  w=self.get(n+'/weights');k=w.shape[-1];pad=(k-1)*d//2
  if s>1:x=F.pad(x,(pad,pad,pad,pad));pad=0
  b=self.get(n+'/biases') if n+'/biases' in self.v else None
  x=F.conv2d(x,w,b,stride=s,padding=pad,dilation=d)
  if norm:x=self.bn(x,n+'/BatchNorm')
  return F.relu(x) if act else x
 @staticmethod
 def resize(x,size):
  # tf.image.resize_bilinear(align_corners=False,half_pixel_centers=False).
  ih,iw=x.shape[-2:];oh,ow=size
  yy=(torch.arange(oh,dtype=x.dtype)*ih/oh).clamp(max=ih-1);xx=(torch.arange(ow,dtype=x.dtype)*iw/ow).clamp(max=iw-1)
  y0=yy.long();y1=(y0+1).clamp(max=ih-1);x0=xx.long();x1=(x0+1).clamp(max=iw-1)
  wy=(yy-y0)[None,None,:,None];wx=(xx-x0)[None,None,None,:]
  x=x.index_select(2,y0)*(1-wy)+x.index_select(2,y1)*wy
  return x.index_select(3,x0)*(1-wx)+x.index_select(3,x1)*wx
 @torch.inference_mode()
 def __call__(self,x):
  size=x.shape[-2:];x=(x-.5)/.25
  x=self.conv(x,'resnet_v2_50/conv1',s=2,norm=False,act=False)
  h,w=x.shape[-2:];ph=max((h+1)//2*2+1-h,0);pw=max((w+1)//2*2+1-w,0)
  x=F.max_pool2d(F.pad(x,(pw//2,pw-pw//2,ph//2,ph-ph//2),value=float('-inf')),3,2)
  low=None
  for block,units in enumerate([3,4,6,3],1):
   for unit in range(1,units+1):
    n=f'resnet_v2_50/block{block}/unit_{unit}/bottleneck_v2';s=2 if block<=2 and unit==units else 1;d=2 if block==4 else 1
    pre=F.relu(self.bn(x,n+'/preact'))
    if n+'/shortcut/weights' in self.v:shortcut=self.conv(pre,n+'/shortcut',s=s,norm=False,act=False)
    else:shortcut=x[:,:,::s,::s]
    z=self.conv(pre,n+'/conv1')
    if block==1 and unit==3:low=z
    z=self.conv(z,n+'/conv2',s=s,d=d);z=self.conv(z,n+'/conv3',norm=False,act=False);x=shortcut+z
  # Original decoder consumes the block4 endpoint before resnet postnorm.
  paths=[self.conv(x,'aspp/conv_1x1')]+[self.conv(x,f'aspp/conv_3x3_{i}',d=d) for i,d in enumerate([6,12,18],1)]
  pooled=self.conv(x.mean((2,3),keepdim=True),'aspp/image_level_features/conv_1x1');paths.append(self.resize(pooled,x.shape[-2:]))
  x=self.conv(torch.cat(paths,1),'aspp/conv_1x1_concat');low=self.conv(low,'decoder/low_level_features/conv_1x1')
  x=torch.cat([self.resize(x,low.shape[-2:]),low],1)
  x=self.conv(x,'decoder/upsampling_logits/conv_3x3_1');x=self.conv(x,'decoder/upsampling_logits/conv_3x3_2');x=self.conv(x,'decoder/upsampling_logits/conv_1x1',norm=False,act=False)
  x=self.resize(x,size);return x[:,1:2]-x[:,0:1]
 def eval(self):return self

if __name__=='__main__':
 import csv,time
 from PIL import Image
 root=Path('/workspace/Temporal-segmentation-model');torch.set_num_threads(4);net=DeepLabNPZ(root/'02_root_background/assets/weights/deeplab');records=[]
 rows=list(csv.DictReader(open(root/'02_root_background/assets/data/manifest.csv',encoding='utf-8-sig')))
 for r in [r for r in rows if r['split']=='test']:
  im=np.array(Image.open(root/'02_root_background/assets'/r['RGB_path']).convert('RGB'));x=torch.from_numpy(im.transpose(2,0,1).copy())[None].float()/255
  p=net(x).sigmoid()[0,0].numpy();q=np.load(root/'02_root_background/assets/test_probabilities/deeplab'/(r['image_id']+'_probability.npy'))
  records.append(dict(image_id=r['image_id'],max_abs=float(np.max(np.abs(p-q))),mean_abs=float(np.mean(np.abs(p-q))),threshold_disagreement=int(((p>=.5)!=(q>=.5)).sum())))
  print(records[-1],flush=True)
 assert all(k.startswith('resnet_v2_50/postnorm/') for k in set(net.v)-net.used),set(net.v)-net.used
 proof=dict(archived_reference='original TF1.12 native256 predictions',images=len(records),max_abs=max(r['max_abs'] for r in records),total_threshold_disagreement=sum(r['threshold_disagreement'] for r in records),records=records)
 Path(__file__).with_name('deeplab_equivalence.json').write_text(json.dumps(proof,indent=2)+'\n')
 assert proof['max_abs']<1e-4 and proof['total_threshold_disagreement']==0,proof
 print('TF112_EQUIVALENCE_PASSED',flush=True)
