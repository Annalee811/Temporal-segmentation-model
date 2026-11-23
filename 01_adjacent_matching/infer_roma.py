"""Full RoMa adjacent observed-date evidence; never warp final RGB nonrigidly."""
import argparse,gc,hashlib,json,os,time
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
from romatch import roma_outdoor

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--positions',nargs='*');p.add_argument('--threads',type=int,default=4);p.add_argument('--extra-pair');p.add_argument('--view',choices=['original','posed'],default='original');a=p.parse_args()
    torch.set_num_threads(a.threads);cv2.setNumThreads(1);torch.manual_seed(20261019);torch.set_float32_matmul_precision('highest')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('LOAD_FULL_ROMA',device,flush=True)
    model=roma_outdoor(device=device,coarse_res=560,upsample_res=864,symmetric=True,use_custom_corr=False)
    pins=Path(__file__).with_name('model_provenance.json')
    if pins.exists():
        for weight in json.loads(pins.read_text())['weights']:
            checkpoint=Path(torch.hub.get_dir())/'checkpoints'/weight['file']
            assert sha(checkpoint)==weight['sha256'],('RoMa checkpoint checksum mismatch',weight['file'])
    dirs=sorted((a.work/'inputs/raw').iterdir(),key=lambda p:p.name)
    for folder in dirs:
        if a.positions and folder.name not in a.positions:continue
        images={int(p.stem[1:]):p for p in folder.iterdir() if p.is_file()};dates=sorted(images)
        if a.view=='posed':
            images={int(p.name[1]):p for p in (a.work/'deliverable'/folder.name/'RGB').glob('T*_independent_stitched_RGB.png')};dates=sorted(images)
        if len(dates)<2:continue
        pairs=[tuple(int(x[1:]) for x in a.extra_pair.split('_to_'))] if a.extra_pair else list(zip(dates,dates[1:]))
        for t,s in pairs:
            out=a.work/('posed_roma_evidence' if a.view=='posed' else 'roma_evidence')/folder.name/f'T{t}_to_T{s}';out.mkdir(parents=True,exist_ok=True)
            hashes=[sha(images[t]),sha(images[s])]
            if (out/'completed.json').exists():
                assert json.loads((out/'completed.json').read_text())['input_sha256']==hashes
                print('REUSE',folder.name,t,s,flush=True);continue
            start=time.time();A=Image.open(images[t]).convert('RGB');B=Image.open(images[s]).convert('RGB')
            print('MATCH_START',folder.name,t,s,A.size,B.size,flush=True)
            with torch.inference_mode():warp,certainty=model.match(A,B,device=device)
            if warp.ndim==4:warp,certainty=warp[0],certainty[0]
            w=warp.cpu().float().numpy();c=certainty.cpu().float().numpy();mid=w.shape[1]//2

            np.savez_compressed(out/'dense_evidence.npz',forward=w[:,:mid,2:].astype('float16'),backward=w[:,mid:,:2].astype('float16'),source_confidence=c[:,:mid].astype('float16'),target_confidence=c[:,mid:].astype('float16'),source_wh=A.size,target_wh=B.size)
            yy,xx=np.mgrid[:w.shape[0]:8,:mid:8];xy=np.stack([(xx+.5)*A.width/mid-.5,(yy+.5)*A.height/w.shape[0]-.5],-1)
            target=(w[yy,xx,2:]+1)*[B.width/2,B.height/2]-.5
            np.savez_compressed(out/'sampled_matches.npz',source_xy=xy.reshape(-1,2),target_xy=target.reshape(-1,2),source_confidence=c[yy,xx].ravel())
            meta=dict(position=folder.name,source_period=t,target_period=s,input_sha256=hashes,source_wh=A.size,target_wh=B.size,device=str(device),model='RoMa outdoor (full model)',coarse_resolution=560,upsample_resolution=864,symmetric=True,seed=20261019,compact_fields_dtype='float16',input_view=a.view,elapsed_seconds=time.time()-start)
            (out/'completed.json').write_text(json.dumps(meta,indent=2));print('MATCH_DONE',folder.name,t,s,round(meta['elapsed_seconds'],2),flush=True)
            del warp,certainty,w,c,A,B;gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
    if a.view=='posed' and not a.positions:
        (a.work/'inference_complete.json').write_text(json.dumps(dict(input_view='posed',all_eligible_positions_completed=True)))

if __name__=='__main__':main()
