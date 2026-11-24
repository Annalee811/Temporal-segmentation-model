"""Native-size binary root inference for a V35 U-Net checkpoint."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from models import RootUNet

def predict_image(model,rgb,batch_size=2):
 h,w=rgb.shape[:2];ph=max(256,h);pw=max(256,w);im=np.pad(rgb,((0,ph-h),(0,pw-w),(0,0)),mode='edge')
 xs=sorted(set(list(range(0,pw-255,192))+[pw-256]));ys=sorted(set(list(range(0,ph-255,192))+[ph-256]));coords=[(x,y) for y in ys for x in xs]
 win=np.maximum(np.hanning(256)[:,None]*np.hanning(256)[None,:],.02).astype('float32');sm=np.zeros((ph,pw),np.float32);wg=np.zeros_like(sm)
 model.eval()
 with torch.inference_mode():
  for i in range(0,len(coords),batch_size):
   cc=coords[i:i+batch_size];x=torch.from_numpy(np.stack([im[y:y+256,x:x+256] for x,y in cc]).transpose(0,3,1,2).copy()).float()/255
   for (x,y),p in zip(cc,model(x).sigmoid()[:,0].numpy()):sm[y:y+256,x:x+256]+=p*win;wg[y:y+256,x:x+256]+=win
 return (sm/wg)[:h,:w]

def main():
 p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--image',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--threads',type=int,default=4);p.add_argument('--threshold',type=float,default=.5);a=p.parse_args();torch.set_num_threads(a.threads)
 net=RootUNet();net.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=True)['state_dict']);rgb=np.array(Image.open(a.image).convert('RGB'));prob=predict_image(net,rgb);a.output.mkdir(parents=True,exist_ok=True)
 np.save(a.output/(a.image.stem+'_root_probability.npy'),prob);Image.fromarray((prob>=a.threshold).astype('uint8')*255).save(a.output/(a.image.stem+'_root_mask.png'))
if __name__=='__main__':main()
