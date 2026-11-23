"""Recompute V66 from released RGB/binary predictions and the original head weights."""
import argparse,json,sys,subprocess,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'core'))
import common as c

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['infer','train']);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--binary-root',type=Path,default=HERE/'assets/frozen_binary')
    ap.add_argument('--models',nargs='+',choices=c.MODELS,default=c.MODELS);ap.add_argument('--threads',type=int,default=2)
    a=ap.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    assets=HERE/'assets';a.binary_root=a.binary_root.resolve()
    if a.mode=='infer':torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    rows=[]
    for m in a.models:
        gates={t:np.array(Image.open(a.binary_root/m/'upper'/f'{t}_root_mask.png'))>0 for t in c.PERIODS}
        rgb={t:np.array(Image.open(assets/'RGB'/f'{t}_RGB.png')) for t in c.PERIODS}
        cache=a.output/'feature_cache'/m;cache.mkdir(parents=True,exist_ok=True)
        for prev,t in zip(c.PERIODS,c.PERIODS[1:]):
            p=cache/(t+'.npy')
            if not p.exists():
                f=c.old.features(rgb[prev],rgb[t],np.load(a.binary_root/m/'upper'/f'{prev}_probability.npy'),np.load(a.binary_root/m/'upper'/f'{t}_probability.npy'),gates[prev],gates[t])
                np.save(p,f.astype(np.float16));del f
            print('FEATURE',m,t,flush=True)
        if a.mode=='train':continue
        ck=torch.load(assets/'weights'/m/'best_temporal_head.pt',map_location='cpu',weights_only=True)
        net=c.AxisHead().to(memory_format=torch.channels_last);net.load_state_dict(ck['state_dict']);net.eval()
        out=a.output/m/'full';out.mkdir(parents=True,exist_ok=True)
        for t in c.PERIODS:
            tic=time.monotonic()
            if t=='T1':pred=np.where(gates[t],4,0).astype(np.uint8)
            else:
                prob=c.predict_probability(net,np.load(cache/(t+'.npy'),mmap_mode='r'),torch.device('cpu'))
                np.savez_compressed(out/f'{t}_probability.npz',probability=prob.astype(np.float16))
                pred=np.where(gates[t],prob.argmax(0)+1,0).astype(np.uint8)
            Image.fromarray(pred).save(out/f'{t}_classes.png')
            expected=np.array(Image.open(assets/'expected'/m/f'{t}_classes.png'))
            changed=int((pred!=expected).sum());assert np.array_equal(pred>0,gates[t])
            rows.append(dict(model=m,period=t,changed_pixels_from_archived_V66=changed,seconds=time.monotonic()-tic))
            print('INFER',m,t,'different_pixels',changed,'seconds',round(time.monotonic()-tic,1),flush=True)
    if a.mode=='train':
        import shutil
        for t in c.PERIODS:
            for sub,src in [('upper',assets/'RGB'/f'{t}_RGB.png'),('current',assets/'training_targets'/f'{t}_classes.png')]:
                p=a.output/'data'/sub/src.name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,p)
        for n in ['paired_manifest.csv','spatial_manifest.csv']:
            shutil.copy2(assets/n,a.output/'data'/n)
        annotation_manifest=assets/'training_annotation_manifest.json'
        shutil.copy2(annotation_manifest,a.output/'data/active_progress.json')
        prep=json.loads((assets/'training_parameters.json').read_text())
        training_roots=a.output/'binary_inputs'
        for m in a.models:
            folder=training_roots/m;folder.mkdir(parents=True,exist_ok=True)
            if not (folder/'upper').exists():(folder/'upper').symlink_to(a.binary_root/m/'upper',target_is_directory=True)
            shutil.copy2(a.binary_root/m/'input_manifest.json',folder/'input_manifest.json')
        prep['roots']=str(training_roots)
        prep['immutable_inputs']=[dict(path=str(annotation_manifest),sha256=c.sha(annotation_manifest))]

        for m in a.models:
            folder=a.output/m;folder.mkdir(exist_ok=True)
        c.js(a.output/'preparation.json',prep)
        for m in a.models:subprocess.run([sys.executable,str(HERE/'core/train.py'),'--run',str(a.output),'--model',m,'--threads',str(a.threads)],check=True)
        return
    c.js(a.output/'reproduction_check.json',dict(passed=all(r['changed_pixels_from_archived_V66']==0 for r in rows),actual_model_forward=True,features_regenerated_from_RGB_and_binary=True,models=a.models,rows=rows))
    assert all(r['changed_pixels_from_archived_V66']==0 for r in rows),'Check arithmetic/library versions; expected predictions were never used to generate output.'

if __name__=='__main__':main()
