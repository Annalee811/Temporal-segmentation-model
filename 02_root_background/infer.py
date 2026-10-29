"""V35 selected binary models; exact internal test protocol or full-image inference."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import common as c
HERE=Path(__file__).resolve().parent

def predictor(model,threads):
    folder=HERE/'assets/weights'/model;sel=c.read(folder/'selection.json')
    ck=folder/('best_model.npz' if model=='deeplab' else 'best_model.pt')
    assert c.sha(ck)==sel['checkpoint_SHA256'],'Checkpoint changed'
    if model!='deeplab':
        import torch
        from models import create
        torch.set_num_threads(threads);net=create(model)
        net.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True)['state_dict']);net.eval()
        def infer(ims):
            with torch.inference_mode():return net(torch.from_numpy(ims.transpose(0,3,1,2).copy())).sigmoid()[:,0].numpy()
        return infer,lambda:None
    import tensorflow as tf
    assert tf.__version__.startswith('1.12'),'Use the supplied Python3.5/TF1.12 container.'
    sys.path.insert(0,str(HERE/'vendor/deeplab'));import deeplab_model
    x=tf.placeholder(tf.float32,[None,256,256,3])
    z=deeplab_model.deeplab_v3_plus_generator(2,16,'resnet_v2_50',None,.90)((x-.5)/.25,False)
    prob=tf.sigmoid(z[:,:,:,1]-z[:,:,:,0]);variables=tf.global_variables()
    ph=[tf.placeholder(v.dtype.base_dtype,v.shape) for v in variables];ops=[tf.assign(v,q) for v,q in zip(variables,ph)]
    sess=tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=threads,inter_op_parallelism_threads=1))
    sess.run(tf.global_variables_initializer())
    with np.load(str(ck)) as saved:
        names=c.read(folder/'weight_names.json');values={n:saved['v%04d'%i] for i,n in enumerate(names)}
        assert set(v.name for v in variables)==set(values),'Tensor names differ from original graph'
        sess.run(ops,{q:values[v.name] for v,q in zip(variables,ph)})
    return lambda ims:sess.run(prob,{x:ims}),sess.close

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=c.MODELS,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--evaluate',action='store_true');p.add_argument('--cached',action='store_true');p.add_argument('--threads',type=int,default=2);p.add_argument('--batch',type=int,default=2);a=p.parse_args()
    assert not a.cached or a.evaluate,'Cached mode is only for test-probability metric verification'
    a.output.mkdir(parents=True,exist_ok=True)
    if a.cached:infer=None;close=lambda:None
    else:infer,close=predictor(a.model,a.threads)
    try:
        if not a.evaluate:
            c.full_predict(infer,HERE/'assets',a.output,.5,a.batch);return
        data=c.Data(HERE/'assets');rows=[r for r in data.rows if r['split']=='test'];assert len(rows)==24
        if a.cached:maps={r['image_id']:np.load(str(HERE/'assets/test_probabilities'/a.model/(r['image_id']+'_probability.npy'))) for r in rows}
        else:maps=data.predict(infer,'test',a.batch)
        measured,per=data.evaluate(maps,.5)
        reference=next(r for r in c.csvread(HERE/'assets/reference_metrics.csv') if r['model']==c.NAMES[a.model])
        measured.update(model=c.NAMES[a.model],threshold=.5,views=24,selected_epoch=int(reference['selected_epoch']),actual_model_forward=not a.cached)
        measured['passed']=all(measured[k]==int(reference[k]) for k in ['TP','FP','FN','TN'])
        c.js(a.output/'measured_metrics.json',measured);c.csvout(a.output/'per_tile_metrics.csv',per)
        print(json.dumps(measured),flush=True)
        assert measured['passed'],'Test counts changed; archived probabilities are not substituted for model outputs'
    finally:close()
if __name__=='__main__':main()
