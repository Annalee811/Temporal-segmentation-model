"""Predict the66 preserved common-field RGBs with the selected full250 V35."""
import argparse, csv, hashlib, json, os, shutil, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from models import RootUNet
from infer_full_250 import predict_image

ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def js(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp')
    q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
def csvout(p,rr):
    with Path(p).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def immutable_copy(src,dst):
    src,dst=Path(src),Path(dst)
    if dst.exists():assert sha(src)==sha(dst);return
    try:os.link(src,dst)
    except OSError:shutil.copy2(src,dst)

def main(a):
    source=a.source.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    done=json.load(open(a.training/'completion.json'));assert done['epochs']==100 and done['supplemental_training_patches']==264
    weights=a.training/'weights/V35_full250_best.pt';model_sha=sha(weights);assert model_sha==done['best_weights_SHA256']
    original=list(csv.DictReader(open(source/'prediction_manifest.csv',encoding='utf-8-sig')))
    assert len(original)==66 and len({r['position'] for r in original})==18
    torch.set_num_threads(4);net=RootUNet().to(memory_format=torch.channels_last)
    net.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True)['state_dict']);net.eval()
    records=[];start=time.monotonic()
    for r in original:
        pos,t=r['position'],r['period'];src=source/pos;dst=out/pos;dst.mkdir(exist_ok=True)
        for name in ['common_domain.png','geometry_metadata.json']:
            immutable_copy(src/name,dst/name)
        rgb_path=src/(t+'_common_RGB.png');assert sha(rgb_path)==r['source_common_RGB_SHA256']
        immutable_copy(rgb_path,dst/rgb_path.name);rgb=np.array(Image.open(rgb_path).convert('RGB'))
        domain=np.array(Image.open(dst/'common_domain.png'))>0;assert rgb.shape[:2]==domain.shape
        js(out/'status.json',dict(state='binary_prediction',completed_images=len(records),total_images=66,position=pos,period=t))
        meta_path=dst/(t+'_metadata.json')
        if meta_path.exists():
            meta=json.load(open(meta_path));assert meta['model_SHA256']==model_sha and meta['source_common_RGB_SHA256']==r['source_common_RGB_SHA256']
            for suffix,key in [('_root_mask.png','root_mask_SHA256'),('_root_probability.npz','probability_SHA256')]:assert sha(dst/(t+suffix))==meta[key]
            records.append(meta);continue
        p=predict_image(net,rgb);assert p.shape==domain.shape and np.isfinite(p).all()
        p=np.where(domain,p,0).astype('float32');mask=p>=.5;assert mask.any() and not mask[~domain].any()
        Image.fromarray(mask.astype('uint8')*255).save(dst/(t+'_root_mask.png'))
        np.savez_compressed(dst/(t+'_root_probability.npz'),root_probability=p,common_domain=domain.astype('uint8'))
        overlay=rgb.copy();overlay[mask]=(rgb[mask]*.25+np.array([255,193,7])*.75).astype('uint8')
        assert np.array_equal(overlay[~mask],rgb[~mask]);Image.fromarray(overlay).save(dst/(t+'_root_overlay.png'))
        meta=dict(position=pos,material=r['material'],period=t,width=rgb.shape[1],height=rgb.shape[0],shared_domain_pixels=int(domain.sum()),predicted_root_pixels=int(mask.sum()),
            source_common_RGB_SHA256=r['source_common_RGB_SHA256'],model_SHA256=model_sha,training_epochs=100,selected_epoch=done['selected_epoch'],
            training_patches=1048,supplemental_training_patches=264,binary_window=250,binary_stride=188,threshold=.5,
            additional_RGB_resizing=False,additional_registration=False,geometric_mask_applied=True,
            domain_mask_SHA256=sha(dst/'common_domain.png'),root_mask_SHA256=sha(dst/(t+'_root_mask.png')),
            probability_SHA256=sha(dst/(t+'_root_probability.npz')),overlay_SHA256=sha(dst/(t+'_root_overlay.png')))
        js(meta_path,meta);records.append(meta);csvout(out/'prediction_manifest.csv',records)
        print('FULL250_COMMON_PREDICTED',len(records),pos,t,int(mask.sum()),flush=True)
    assert len(records)==66
    csvout(out/'prediction_manifest.csv',records)
    for name in ['missing_periods.csv','excluded_positions.csv']:shutil.copy2(source/name,out/name)
    js(out/'completion.json',dict(state='complete',positions=18,images=66,model_checkpoint='V35_full250_best.pt',model_SHA256=model_sha,
        training_epochs=100,selected_epoch=done['selected_epoch'],training_patches=1048,supplemental_training_patches=264,binary_window=250,binary_stride=188,
        threshold=.5,scope='Unchanged common RGB and geometry; actual observed periods only',missing_images_synthesized=False,
        RGB_resampled=False,seconds=time.monotonic()-start))
    js(out/'status.json',dict(state='complete',completed_images=66,total_images=66))
    print('ALL66_FULL250_COMMON_PREDICTIONS_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=ROOT/'work/V35_four_period_common')
    p.add_argument('--training',type=Path,default=ROOT/'work/V35_full250_all66')
    p.add_argument('--output',type=Path,default=ROOT/'work/V35_full250_common66');main(p.parse_args())
