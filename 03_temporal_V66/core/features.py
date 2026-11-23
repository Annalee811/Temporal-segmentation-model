from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from junctions import extra_features
SIDE=256
PERIODS=['T1','T2','T3','T4']
NAMES={1:'persistent',2:'newly_observed',3:'tip_growth'}

def coords(h,w,side=SIDE):
    assert min(h,w)>=side
    xs=list(range(0,w-side+1,side//2));ys=list(range(0,h-side+1,side//2))
    if xs[-1]!=w-side:xs.append(w-side)
    if ys[-1]!=h-side:ys.append(h-side)
    return [(x,y) for y in ys for x in xs]

def features(previous,current,pp,cp,previous_mask,current_mask):
    """Only RGB and model predictions. Never consumes a human root mask."""
    binary=previous_mask.astype(bool);sk=skeletonize(binary);degree=ndi.convolve(sk.astype(np.uint8),np.ones((3,3),np.uint8),mode='constant')-sk
    cues=[]
    for points in [sk&(degree==1),sk&(degree>=3)]:
        cues.append(np.exp(-ndi.distance_transform_edt(~points)**2/200).astype(np.float32) if points.any() else np.zeros(pp.shape,np.float32))
    distance=np.minimum(ndi.distance_transform_edt(~binary),32).astype(np.float32)/32 if binary.any() else np.ones(pp.shape,np.float32)

    base=np.concatenate([previous.astype(np.float32)/255*binary[...,None],current.astype(np.float32)/255*current_mask[...,None],(pp*binary)[...,None],(cp*current_mask)[...,None],cues[0][...,None],cues[1][...,None],distance[...,None]],2)
    return np.concatenate([base,extra_features(binary,current_mask.astype(bool))],2)

def predict(model,feature,device,batch=4):
    import torch
    h,w=feature.shape[:2];xy=coords(h,w);blend=np.maximum(np.outer(np.hanning(SIDE),np.hanning(SIDE)),.05).astype(np.float32)
    total=np.zeros((3,h,w),np.float32);weight=np.zeros((h,w),np.float32);model.eval()
    with torch.inference_mode():
        for i in range(0,len(xy),batch):
            cc=xy[i:i+batch];ims=np.stack([feature[y:y+SIDE,x:x+SIDE].transpose(2,0,1) for x,y in cc])
            xx=torch.from_numpy(ims.copy()).to(device).contiguous(memory_format=torch.channels_last)
            enabled=device.type=='cpu' and torch.cpu._is_avx512_bf16_supported()
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=enabled):logits=model(xx)
            p=logits.float().softmax(1).cpu().numpy()
            for (x,y),v in zip(cc,p):total[:,y:y+SIDE,x:x+SIDE]+=v*blend;weight[y:y+SIDE,x:x+SIDE]+=blend
    assert (weight>0).all()
    result=total/weight[None]
    if hasattr(model,'class_multipliers'):

        result=result.astype(np.float16).astype(np.float32)*np.asarray(model.class_multipliers,np.float32)[:,None,None]
        result/=np.maximum(result.sum(0,keepdims=True),1e-12)
    if hasattr(model,'class_adjustment'):
        result*=np.asarray(model.class_adjustment,np.float32)[:,None,None]
        result/=np.maximum(result.sum(0,keepdims=True),1e-12)
    return result

def confusion(gt,pred,root_only=False):
    valid=(gt!=255)&((gt>0) if root_only else True)
    g=gt[valid].astype(np.int64);p=pred[valid].astype(np.int64)
    return np.bincount(g*4+p,minlength=16).reshape(4,4)

def metrics(cm):
    rows=[]
    for k in [1,2,3]:
        tp=int(cm[k,k]);fp=int(cm[:,k].sum()-tp);fn=int(cm[k].sum()-tp);support=int(cm[k].sum())
        rows.append(dict(class_id=k,name=NAMES[k],support_pixels=support,TP=tp,FP=fp,FN=fn,
            Precision=tp/(tp+fp) if tp+fp else None,Recall=tp/support if support else None,
            Dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,IoU=tp/(tp+fp+fn) if tp+fp+fn else None))
    return dict(classes=rows,macro_Dice=float(np.mean([r['Dice'] for r in rows if r['Dice'] is not None])) if any(r['Dice'] is not None for r in rows) else None,confusion_matrix=cm.tolist())
