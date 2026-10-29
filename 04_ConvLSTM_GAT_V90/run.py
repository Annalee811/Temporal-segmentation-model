"""Recompute learned graph descriptors and the exact structural/reviewed V90 result."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['infer','train']);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    work=a.output.resolve();work.mkdir(parents=True,exist_ok=True)

    for source in (HERE/'assets').iterdir():
        if source.name.startswith('expected') or source.name in ['review_support.png','review_annotation.json']:continue
        dest=work/source.name
        if not dest.exists():
            if source.is_dir():shutil.copytree(source,dest)
            else:shutil.copy2(source,dest)
    os.environ['ROOT_REPRO_WORK']=str(work)
    sys.path.insert(0,str(HERE/'core'))
    if a.mode=='train':
        for script in ['train.py','joint_train.py']:subprocess.run([sys.executable,str(HERE/'core'/script)],check=True)

    import torch
    from model import LocalCorrector
    from data import Data
    from backbone import pool_groups
    from train import PairHead
    from paths import js
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    data=Data();pairdata=dict(np.load(work/'joint/data/pairs.npz'))
    import pickle
    with (work/'joint/data/topology.pkl').open('rb') as f:top=pickle.load(f)
    ck=torch.load(work/'joint/joint_best.pt',map_location='cpu',weights_only=False)
    model=LocalCorrector();model.load_state_dict(ck['backbone']);model.eval();pair=PairHead(pairdata['x'].shape[1]);pair.load_state_dict(ck['pair']);pair.eval()
    groups={(g['period'],g['hub']):g for g in top['groups']};nodes=[];ptr=[0];A=np.zeros(len(pairdata['x']),np.int64);B=A.copy()
    for t,items in top['arms'].items():
        for hub,arms in items:
            ids=[]
            for arm in arms:ids.append(len(ptr)-1);nodes.extend(arm['nodes']);ptr.append(len(nodes))
            for key,k in groups[t,hub]['pair_indices'].items():i,j=map(int,key.split(','));A[k]=ids[i];B[k]=ids[j]
    with torch.inference_mode():
        h,links,z=data.forward(model);e=torch.nn.functional.normalize(pool_groups(z,torch.tensor(nodes),torch.tensor(ptr)),dim=1)
        x=torch.cat([(e[A]-e[B]).abs(),e[A]*e[B],torch.from_numpy(pairdata['raw'][:,96:])],1)
        x=(x-torch.from_numpy(pairdata['mean']))/torch.from_numpy(pairdata['std']);logit=pair(x).numpy()
    learned=dict(h=h.numpy(),embedding=z.numpy(),connection=links.sigmoid().numpy(),features=x.numpy())
    np.save(work/'joint/joint_pair_logits.npy',logit);np.savez_compressed(work/'joint/joint_inference.npz',**learned)
    expected=np.load(HERE/'assets/expected_joint_features.npz');differences={k:float(np.abs(learned[k]-expected[k]).max()) for k in learned}
    differences['pair_logits']=float(np.abs(logit-np.load(HERE/'assets/expected_pair_logits.npy')).max())
    print('LEARNED_FORWARD_MAX_DIFFERENCES',differences,flush=True)
    for script in ['decode.py','branch_review.py','axis_consistency.py','junction_consistency.py']:
        subprocess.run([sys.executable,str(HERE/'core'/script)],check=True)
        if script=='branch_review.py':
            shutil.copytree(work/'branch/candidate_full',work/'branch/full',dirs_exist_ok=True)

    review=json.loads((HERE/'assets/review_annotation.json').read_text());review_mask=np.array(Image.open(HERE/'assets/review_support.png'))>0
    full=work/'reviewed/full';full.mkdir(parents=True,exist_ok=True);rows=[]
    for t in ['T1','T2','T3','T4']:
        pred=np.array(Image.open(work/'junction/full'/f'{t}_classes.png'));before=pred.copy()
        if t==review['period']:
            target=review_mask&(pred>0);assert int(target.sum())==review['foreground_pixels'];pred[target]=review['class_code']
        assert np.array_equal(pred>0,before>0)
        Image.fromarray(pred).save(full/f'{t}_classes.png')
        expected=np.array(Image.open(HERE/'assets/expected'/f'{t}_classes.png'))
        rows.append(dict(period=t,different_pixels_from_archived_V90=int((pred!=expected).sum())))
    js(work/'reproduction_check.json',dict(passed=all(r['different_pixels_from_archived_V90']==0 for r in rows),actual_ConvLSTM_GAT_forward=True,
        full_structural_decoding_recomputed=True,explicit_review_supervision_applied=True,neural_numeric_differences=differences,rows=rows))
    print('FINAL_COMPARISON',rows,flush=True)
    assert all(r['different_pixels_from_archived_V90']==0 for r in rows),'No saved final prediction is used to fill or overwrite a mismatch.'

if __name__=='__main__':main()
