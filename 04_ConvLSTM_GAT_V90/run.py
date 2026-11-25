"""Run ConvLSTM and GAT with automatic structural refinement."""
import argparse,os,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['infer','train']);p.add_argument('--output',type=Path,required=True);
    p.add_argument('--legacy-three-class', action='store_true', help='Export three-class structural predictions')
    p.add_argument('--ablation', choices=['no-temporal-memory', 'no-gat'])
    p.add_argument('--common-source', type=Path, help='Latest V35 common-field run; uses the updated foreground and V90 transfer inference')
    p.add_argument('--transfer-runtime', type=Path, default=HERE.parent/'work/V90_full250_common66/runtime')
    a=p.parse_args()
    if a.common_source is not None:
        if a.mode != 'infer' or a.ablation or a.legacy_three_class:
            p.error('The updated common-field pipeline requires infer and exports two root classes')
        subprocess.run([sys.executable, str(HERE/'predict_latest_common.py'), '--source', str(a.common_source), '--runtime', str(a.transfer_runtime), '--output', str(a.output)], check=True)
        return
    if a.ablation and a.mode != 'infer':
        p.error('--ablation requires infer')
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
    from paths import PERIODS, js, sha
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    data=Data();pairdata=dict(np.load(work/'joint/data/pairs.npz'))
    import pickle
    with (work/'joint/data/topology.pkl').open('rb') as f:top=pickle.load(f)
    ck=torch.load(work/'joint/joint_best.pt',map_location='cpu',weights_only=False)
    model=LocalCorrector();model.load_state_dict(ck['backbone']);model.eval();pair=PairHead(pairdata['x'].shape[1]);pair.load_state_dict(ck['pair']);pair.eval()
    model.use_temporal_memory = a.ablation != 'no-temporal-memory'
    model.use_graph_attention = a.ablation != 'no-gat'
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
    differences = None
    if not a.ablation:
        with np.load(HERE/'assets/expected_joint_features.npz') as expected:
            differences = {k:float(np.abs(learned[k]-expected[k]).max()) for k in learned}
        differences['pair_logits'] = float(np.abs(logit-np.load(HERE/'assets/expected_pair_logits.npy')).max())
        print('LEARNED_FORWARD_MAX_DIFFERENCES',differences,flush=True)
    for script in ['decode.py','branch_review.py','axis_consistency.py','junction_consistency.py']:
        subprocess.run([sys.executable,str(HERE/'core'/script)],check=True)
        if script=='branch_review.py':
            shutil.copytree(work/'branch/candidate_full',work/'branch/full',dirs_exist_ok=True)

    predicted = work / 'prediction/full'
    predicted.mkdir(parents=True, exist_ok=True)
    rows = []
    for period in PERIODS:
        filename = f'{period}_classes.png'
        with Image.open(work / 'junction/full' / filename) as image:
            structural = np.asarray(image)
        pred = structural if a.legacy_three_class else model.temporal_classes(structural)
        allowed = [0, 4] if period == 'T1' else [0, 1, 2, 3] if a.legacy_three_class else [0, 1, 2]
        assert np.isin(pred, allowed).all()
        v35_support = np.asarray(Image.open(work / 'base/unet/full' / filename)) > 0
        assert np.array_equal(structural > 0, v35_support), f'V90 changed upstream root foreground: {period}'
        assert np.array_equal(pred > 0, v35_support), f'Temporal labels left the root foreground: {period}'
        output = predicted / filename
        Image.fromarray(pred).save(output)
        rows.append(dict(period=period, root_pixels=int(np.count_nonzero(pred)), sha256=sha(output)))
    classes = {1: 'persistent', 2: 'new_root', 3: 'extension'} if a.legacy_three_class else model.CLASS_CODES
    js(work / 'inference.json', dict(
        model='Temporal Root Graph Decoding (TRGD)',
        inference_ablation=a.ablation,
        checkpoint_sha256=sha(work/'joint/joint_best.pt'),
        actual_ConvLSTM_GAT_forward=True,
        full_structural_decoding_recomputed=True,
        foreground_contract='Temporal foreground equals frozen upstream root support',
        class_codes={0: 'background', **classes, 4: 'T1_baseline'},
        neural_numeric_differences=differences,
        rows=rows,
    ))
    print('PREDICTIONS', predicted, flush=True)

if __name__=='__main__':main()
