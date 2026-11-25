"""Native-size binary V35 inference using the full250 training window contract."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from models import RootUNet


def predict_image(model, rgb, batch_size=8):
    h,w=rgb.shape[:2];ph,pw=max(h,250),max(w,250)
    im=np.pad(rgb,((0,ph-h),(0,pw-w),(0,0)),mode='edge')
    xs=sorted(set(list(range(0,pw-249,188))+[pw-250]))
    ys=sorted(set(list(range(0,ph-249,188))+[ph-250]))
    coords=[(x,y) for y in ys for x in xs]
    win=np.maximum(np.hanning(250)[:,None]*np.hanning(250)[None,:],.02).astype('float32')
    sm=np.zeros((ph,pw),np.float32);wg=np.zeros_like(sm);model.eval()
    with torch.inference_mode():
        for i in range(0,len(coords),batch_size):
            cc=coords[i:i+batch_size]
            x=torch.from_numpy(np.stack([im[y:y+250,x:x+250] for x,y in cc]).transpose(0,3,1,2).copy()).float().to(memory_format=torch.channels_last)/255
            for (x,y),p in zip(cc,model(x).sigmoid()[:,0].numpy()):
                sm[y:y+250,x:x+250]+=p*win;wg[y:y+250,x:x+250]+=win
    assert (wg>0).all()
    return (sm/wg)[:h,:w]


def main(a):
    torch.set_num_threads(a.threads);net=RootUNet().to(memory_format=torch.channels_last)
    net.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=True)['state_dict'])
    rgb=np.array(Image.open(a.image).convert('RGB'));prob=predict_image(net,rgb,a.batch)
    a.output.mkdir(parents=True,exist_ok=True);stem=a.image.stem;mask=prob>=a.threshold
    np.save(a.output/(stem+'_root_probability.npy'),prob)
    Image.fromarray(mask.astype('uint8')*255).save(a.output/(stem+'_root_mask.png'))
    overlay=rgb.copy();overlay[mask]=(rgb[mask]*.25+np.array([240,195,0])*.75).astype('uint8')
    Image.fromarray(overlay).save(a.output/(stem+'_root_overlay.png'))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (a.output/(stem+'_metadata.json')).write_text(json.dumps(dict(width=rgb.shape[1],height=rgb.shape[0],
        classes={'0':'background','255':'all roots'},threshold=a.threshold,window=250,stride=188,
        resized=False,image_SHA256=sha(a.image),weights_SHA256=sha(a.weights)),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threshold',type=float,default=.5);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--batch',type=int,default=8);main(p.parse_args())
