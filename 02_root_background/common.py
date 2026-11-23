"""Native RGB and the frozen current reference; fixed paired split."""
import csv, hashlib, json, random
from pathlib import Path
import numpy as np
from PIL import Image

SEED = 20261021
SIDE = 256
PERIODS = ['T1', 'T2', 'T3', 'T4']
MODELS = ['unet', 'deeplab', 'segformer']
NAMES = {'unet': 'U-Net', 'deeplab': 'DeepLabV3+', 'segformer': 'SegFormer-B0'}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def js(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def read(path): return json.loads(Path(path).read_text())

def csvout(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)

def csvread(path):
    with Path(path).open(encoding='utf-8-sig') as f: return list(csv.DictReader(f))

def counts(gt, pred):
    g = gt == 255; v = gt != 128; p = pred.astype(bool)
    return np.array([(g&p&v).sum(), (~g&p&v).sum(), (g&~p&v).sum(), (~g&~p&v).sum()], np.int64)

def metrics(c):
    tp, fp, fn, tn = map(int, c)
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, Precision=tp/(tp+fp) if tp+fp else None,
                Recall=tp/(tp+fn) if tp+fn else None, Dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                IoU=tp/(tp+fp+fn) if tp+fp+fn else None)

class Data:
    def __init__(self, run):
        self.run = Path(run)
        expected=read(self.run/'data/summary.json')['manifest_SHA256']
        assert sha(self.run/'data/manifest.csv')==expected,'Dataset manifest changed'
        self.rows = csvread(self.run/'data/manifest.csv')
        for r in self.rows:
            for k in ['x','y','width','height','root_pixels','ignored_pixels']: r[k] = int(r[k])
        self.cache = {}

    def load(self, r):
        key = r['image_id']
        if key not in self.cache:
            assert sha(self.run/r['RGB_path'])==r['RGB_sha256'],'Native RGB changed: '+key
            assert sha(self.run/r['mask_path'])==r['mask_sha256'],'Human mask changed: '+key
            self.cache[key] = (np.array(Image.open(self.run/r['RGB_path']).convert('RGB')),
                               np.array(Image.open(self.run/r['mask_path']).convert('L')))
        return self.cache[key]

    def batch(self, rows, augment=False):
        xx=[]; yy=[]; vv=[]
        for r in rows:
            im, gt = self.load(r); im=im.copy(); gt=gt.copy()
            if augment:
                k=random.randrange(4); im=np.rot90(im,k); gt=np.rot90(gt,k)
                if random.random()<.5: im=np.fliplr(im); gt=np.fliplr(gt)
                if random.random()<.5: im=np.flipud(im); gt=np.flipud(gt)
                im=np.clip(im.astype(np.float32)/255*random.uniform(.9,1.1),0,1)**random.uniform(.9,1.1)
            else: im=im.astype(np.float32)/255
            xx.append(im.copy()); yy.append((gt==255).astype(np.float32)); vv.append((gt!=128).astype(np.float32))
        return np.stack(xx), np.stack(yy), np.stack(vv)

    def predict(self, infer, split, batch=2):
        rows=[r for r in self.rows if r['split']==split]; output={}
        for i in range(0,len(rows),batch):
            rr=rows[i:i+batch]; xx,_,_=self.batch(rr)
            for r, p in zip(rr,infer(xx)): output[r['image_id']] = p.astype(np.float32)
        return output

    def evaluate(self, maps, threshold):
        total=np.zeros(4,np.int64); records=[]
        lookup={r['image_id']:r for r in self.rows}
        for key, p in maps.items():
            r=lookup[key]; _,g=self.load(r); cc=counts(g,p>=threshold); total+=cc
            records.append(dict(image_id=key,tile_id=r['tile_id'],period=r['period'],split=r['split'],**metrics(cc)))
        return metrics(total), records

def calibrate(data, maps):
    rows=[]
    for th in np.arange(.1,.901,.025):
        th=float(round(th,3)); m,_=data.evaluate(maps,th); rows.append(dict(threshold=th,**m))
    selected=max(rows,key=lambda r:(r['Dice'],-abs(r['threshold']-.5)))
    return selected,rows

def save_probabilities(folder, maps):
    folder=Path(folder); folder.mkdir(parents=True, exist_ok=True)
    for key,p in maps.items(): np.save(str(folder/(key+'_probability.npy')),p.astype(np.float32))

def full_predict(infer, run, out, threshold, batch=2):
    """Qualitative whole-upper image inference; never used as held-out scoring."""
    run=Path(run); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    han=np.hanning(SIDE).astype(np.float32); blend=np.maximum(han[:,None]*han[None,:],.02)
    for t in PERIODS:
        rgb=np.array(Image.open(run/'data/upper'/(t+'_RGB.png')).convert('RGB')); h,w=rgb.shape[:2]
        xs=list(range(0,w-SIDE+1,192)); ys=list(range(0,h-SIDE+1,192))
        if xs[-1]!=w-SIDE: xs.append(w-SIDE)
        if ys[-1]!=h-SIDE: ys.append(h-SIDE)
        coords=[(x,y) for y in ys for x in xs]; sm=np.zeros((h,w),np.float32); weights=np.zeros_like(sm)
        for i in range(0,len(coords),batch):
            xy=coords[i:i+batch]; ims=np.stack([rgb[y:y+SIDE,x:x+SIDE] for x,y in xy]).astype(np.float32)/255
            for (x,y),p in zip(xy,infer(ims)):
                sm[y:y+SIDE,x:x+SIDE]+=p*blend; weights[y:y+SIDE,x:x+SIDE]+=blend
        assert (weights>0).all(); p=sm/weights
        np.save(str(out/(t+'_probability.npy')),p.astype(np.float32))
        Image.fromarray((p>=threshold).astype(np.uint8)*255).save(str(out/(t+'_root_mask.png')))
        print('FULL_UPPER',t,flush=True)
