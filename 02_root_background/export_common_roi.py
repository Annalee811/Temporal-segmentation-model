"""Crop a V35/V90 preview from existing common-field inference without re-inference."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]

def export(a):
    binary=a.binary_root/a.position;temporal=a.temporal_root/a.position;t=a.period
    rgb=np.array(Image.open(binary/(t+'_common_RGB.png')).convert('RGB'))
    mask=np.array(Image.open(binary/(t+'_root_mask.png')))>0
    classes=np.array(Image.open(temporal/(t+'_classes.png')))
    assert mask.shape==classes.shape==rgb.shape[:2]
    assert np.array_equal(mask,classes>0) and np.isin(classes,[0,1,2]).all()
    h,w=mask.shape;x,y=a.x,a.y;cw,ch=a.width,a.height
    assert x>=0 and y>=0 and cw>0 and ch>0 and x+cw<=w and y+ch<=h
    sl=np.s_[y:y+ch,x:x+cw];im=rgb[sl];m=mask[sl];cl=classes[sl]
    a.output.mkdir(parents=True,exist_ok=True)
    Image.fromarray(im).save(a.output/'common_RGB.png')
    Image.fromarray(m.astype('uint8')*255).save(a.output/'V35_root_mask.png')
    Image.fromarray(cl).save(a.output/'V90_classes.png')
    root_overlay=im.copy();root_overlay[m]=(im[m]*.25+np.array([240,195,0])*.75).astype('uint8')
    Image.fromarray(root_overlay).save(a.output/'V35_root_overlay.png')
    temporal_overlay=im.copy();palette=np.array([[0,0,0],[55,118,185],[230,159,0]],np.uint8)
    temporal_overlay[m]=(im[m]*.25+palette[cl[m]]*.75).astype('uint8')
    Image.fromarray(temporal_overlay).save(a.output/'V90_temporal_overlay.png')
    meta=json.load(open(temporal/(t+'_metadata.json')))
    result=dict(position=a.position,period=t,common_grid_xywh=[x,y,cw,ch],
        source_size_wh=[w,h],crop_resized=False,additional_network_forward=False,
        extraction_policy='Full common-field inference once; crop RGB, binary mask and temporal classes at the same native coordinates',
        binary_weights_SHA256=meta['binary_weights_SHA256'],
        V90_weights_SHA256=meta['V90_weights_SHA256'],
        root_pixels=int(m.sum()),existing_root_pixels=int((cl==1).sum()),new_root_pixels=int((cl==2).sum()),
        V35_vs_V90_foreground_difference_pixels=int((m!=(cl>0)).sum()),
        output_SHA256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.output.glob('*.png')})
    (a.output/'metadata.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary-root',type=Path,default=ROOT/'work/V35_full250_common66')
    p.add_argument('--temporal-root',type=Path,default=ROOT/'work/V90_full250_common66/predictions')
    p.add_argument('--position',required=True);p.add_argument('--period',choices=['T1','T2','T3','T4'],required=True)
    for name in ['x','y','width','height']:p.add_argument('--'+name,type=int,required=True)
    p.add_argument('--output',type=Path,required=True);export(p.parse_args())
