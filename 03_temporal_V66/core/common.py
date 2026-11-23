import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import features as old
from model import TemporalHead,loss
old.MODELS=['unet','deeplab','segformer']
old.MODEL_NAMES={'unet':'U-Net','deeplab':'DeepLabV3+','segformer':'SegFormer-B0'}
old.PALETTE=np.array([[255,255,255],[136,86,167],[0,158,115],[230,171,2],[0,114,178]],np.uint8)
HERE=Path(__file__).resolve().parent
PERIODS=old.PERIODS; MODELS=old.MODELS; MODEL_NAMES=old.MODEL_NAMES
NAMES=old.NAMES; SIDE=256
PALETTE=np.full((256,3),148,np.uint8);PALETTE[:5]=old.PALETTE
AxisHead=TemporalHead
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rows(p):
    with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def csvout(p,v):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(v[0]) if v else ['empty']);w.writeheader();w.writerows(v)
def prob(model,x,amp=False):
    with torch.autocast(device_type=x.device.type,dtype=torch.bfloat16,enabled=amp): z=model(x)
    return z.float().softmax(1)
def predict_probability(model,feature,device,batch=4):return old.predict(model,feature,device,batch)

class Data:
    def __init__(self,run,roots,model):
        self.run=Path(run);self.roots=Path(roots);self.model=model
        self.pairs=rows(self.run/'data/paired_manifest.csv');self.train=[r for r in self.pairs if r['split']=='train']
        self.rgb={t:np.array(Image.open(self.run/'data/upper'/f'{t}_RGB.png')) for t in PERIODS}
        self.gate={t:np.array(Image.open(self.roots/model/'upper'/f'{t}_root_mask.png'))>0 for t in PERIODS}
        self.targets={'current':{t:np.array(Image.open(self.run/'data/current'/f'{t}_classes.png')) for t in PERIODS}}
        self.features={t:np.load(self.run/'feature_cache'/model/(t+'.npy'),mmap_mode='r') for t in PERIODS[1:]}
    def patch(self,row,train=False,rng=None):
        x,y=int(row['x']),int(row['y']);t=row['current']
        f=self.features[t][y:y+SIDE,x:x+SIDE].astype('float32');g=self.targets['current'][t][y:y+SIDE,x:x+SIDE].copy();aug={}
        if train:
            g[~self.gate[t][y:y+SIDE,x:x+SIDE]]=255
            k=int(rng.integers(4));fx=bool(rng.integers(2));fy=bool(rng.integers(2))
            f=np.rot90(f,k);g=np.rot90(g,k)
            if fx:f=np.fliplr(f);g=np.fliplr(g)
            if fy:f=np.flipud(f);g=np.flipud(g)
            b=[float(rng.uniform(.95,1.05)),float(rng.uniform(.95,1.05))]
            for sl,v in zip([slice(0,3),slice(3,6)],b):f[:,:,sl]=np.clip(f[:,:,sl]*v,0,1)
            aug=dict(rotation_quarters=k,flip_x=fx,flip_y=fy,previous_brightness=b[0],current_brightness=b[1])
        return f.transpose(2,0,1).copy(),g.copy(),aug
    def counts(self):
        n=np.zeros(3,np.int64)
        for r in self.train:
            _,g,_=self.patch(r);x,y=int(r['x']),int(r['y']);p=self.gate[r['current']][y:y+SIDE,x:x+SIDE]
            n += [int(((g==k)&p).sum()) for k in [1,2,3]]
        return n
sys.path.insert(0,str(HERE))
