"""Common-root temporal metrics; true background is not an evaluated class."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent
MODELS={'unet':'V66 U-Net','deeplab':'V66 DeepLabV3+','segformer':'V66 SegFormer-B0'}
def measure(folder):
 cm=np.zeros((3,4),np.int64)
 for t in ['T2','T3','T4']:
  g=np.array(Image.open(ROOT/'03_temporal_V66/assets/evaluation_reference'/f'{t}_classes.png'));p=np.array(Image.open(folder/f'{t}_classes.png'))
  use=np.isin(g,[1,2,3]);assert np.isin(p[use],[0,1,2,3]).all();y=g[use].astype(np.int64)-1;z=np.where(p[use]==0,3,p[use]-1)
  cm+=np.bincount(y*4+z,minlength=12).reshape(3,4)
 assert cm.sum()==468203
 tp=np.diag(cm[:,:3]);fp=cm[:,:3].sum(0)-tp;fn=cm.sum(1)-tp
 metrics=dict(Dice=float(np.mean(2*tp/(2*tp+fp+fn))),IoU=float(np.mean(tp/(tp+fp+fn))),Precision=float(np.mean(tp/(tp+fp))),Recall=float(np.mean(tp/(tp+fn))),Accuracy=float(tp.sum()/cm.sum()),missed_root_pixels=int(cm[:,3].sum()),evaluated_root_pixels=int(cm.sum()))
 return metrics,cm.tolist()
def main():
 p=argparse.ArgumentParser();p.add_argument('--v66',type=Path,required=True);p.add_argument('--v90',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 rows=[];matrices={}
 for m,name in MODELS.items():
  metrics,cm=measure(a.v66/m/'full');rows.append(dict(model=name,**metrics));matrices[name]=cm
 if a.v90:
  metrics,cm=measure(a.v90);rows.append(dict(model='V90 reviewed refinement',**metrics));matrices[rows[-1]['model']]=cm
 with (a.output/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (a.output/'confusion.json').write_text(json.dumps(dict(rows=['persistent','new','extension'],columns=['persistent','new','extension','missed'],matrices=matrices),indent=2)+'\n')
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
