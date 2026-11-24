import csv,hashlib,json,random,shutil
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
root=Path('/workspace/Temporal-segmentation-model');base=root/'02_root_background/assets';out=root/'work/V35_310_100epochs';(out/'data/RGB').mkdir(parents=True,exist_ok=True);(out/'data/mask').mkdir(exist_ok=True);(out/'weights').mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvout(p,r):
 with Path(p).open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
rows=[]
for a in list(csv.DictReader(open(base/'data/manifest.csv',encoding='utf-8-sig'))):
 name='original_'+a['image_id']+'.png';shutil.copy2(base/a['RGB_path'],out/'data/RGB'/name);shutil.copy2(base/a['mask_path'],out/'data/mask'/name)
 assert sha(out/'data/RGB'/name)==a['RGB_sha256'];assert sha(out/'data/mask'/name)==a['mask_sha256']
 rows.append(dict(image_id=name,source='original_244',position='0353-2-B3',tube='0353-2',material='0353',period=a['period'],x=int(a['x']),y=int(a['y']),width=int(a['width']),height=int(a['height']),previous_split=a['split'],previous_tile=a['tile_id'],RGB_sha256=a['RGB_sha256'],mask_sha256=a['mask_sha256']))
jpath=root/'work/V35_cross_tube_v2/uploaded_annotation.json';j=json.loads(jpath.read_text());shutil.copy2(jpath,out/'uploaded_annotation.json')
for a in j['images']:
 name=a['name'];p=a['annotation_provenance'];rgb=root/'work/V35_cross_tube_v2/data/RGB'/name;assert sha(rgb)==a['image_sha256'];shutil.copy2(rgb,out/'data/RGB'/name)
 r=np.asarray(a['layers']['mask']);mask=np.repeat(r[::2].astype('uint8'),r[1::2]).reshape(a['height'],a['width']);assert set(np.unique(mask))<={0,128,255};Image.fromarray(mask).save(out/'data/mask'/name)
 rows.append(dict(image_id=name,source='submitted_66',position=p['position'],tube=p['tube'],material=p['material'],period=p['period'],x=int(p['x']),y=int(p['y']),width=500,height=500,previous_split='none_new_partition',previous_tile='',RGB_sha256=sha(rgb),mask_sha256=sha(out/'data/mask'/name)))
assert len(rows)==310
parent=list(range(len(rows)))
def find(i):
 while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
 return i
def union(i,j):parent[find(j)]=find(i)
def overlap(a,b):return a['position']==b['position'] and min(a['x']+a['width'],b['x']+b['width'])>max(a['x'],b['x']) and min(a['y']+a['height'],b['y']+b['height'])>max(a['y'],b['y'])
for i,a in enumerate(rows):
 for k,b in enumerate(rows[:i]):
  if overlap(a,b):union(i,k)
groups={}
for i,r in enumerate(rows):groups.setdefault(find(i),[]).append(r)
keys=sorted(groups,key=lambda k:groups[k][0]['image_id']);seed=20261124;rng=random.Random(seed);best=None
for _ in range(30000):
 shuffled=keys.copy();rng.shuffle(shuffled);picked=[];n=0
 for k in shuffled:
  if n+len(groups[k])<=62:picked.append(k);n+=len(groups[k])
 if n!=62:continue
 val=[r for k in picked for r in groups[k]];nnew=sum(r['source']=='submitted_66' for r in val);nold=62-nnew
 if {r['material'] for r in val if r['source']=='submitted_66'}!={'0353','A017','Q1'}:continue
 score=abs(nnew-14)
 if best is None or score<best[0]:best=(score,set(picked),nnew)
 if score==0:break
assert best is not None
for k,group in groups.items():
 gid='G%03d'%(keys.index(k)+1)
 for r in group:r['group_id']=gid;r['split']='val' if k in best[1] else 'train';m=np.array(Image.open(out/'data/mask'/r['image_id']));r['root_pixels']=int((m==255).sum());r['ignore_pixels']=int((m==128).sum());assert m.shape==(r['height'],r['width'])
for i,a in enumerate(rows):
 for b in rows[:i]:
  if a['split']!=b['split']:assert not overlap(a,b)
assert Counter(r['split'] for r in rows)=={'train':248,'val':62}
csvout(out/'data/manifest.csv',rows)
original=base/'weights/unet/best_model.pt';shutil.copy2(original,out/'weights/V35_original_epoch77.pt')
audit=dict(images=310,original_images=244,submitted_images=66,split='248 train / 62 validation; no test split',source_counts={s:dict(Counter(r['split'] for r in rows if r['source']==s)) for s in ['original_244','submitted_66']},groups=len(groups),split_seed=seed,group_rule='Connected components of spatially overlapping windows at the same observation position, across all dates and translated variants; no overlapping native pixels across train and validation',new_binary_labels_restored_from_uploaded_JSON=True,old_validation_test_splits_discarded=True,old_holdout_ignore_guard_removed=True,original_weights_SHA256=sha(original),annotation_SHA256=sha(jpath),classes={'0':'background','255':'all roots','128':'ignore annotation only'},temporal_layers_used=False,manifest_SHA256=sha(out/'data/manifest.csv'),new_reference='User submitted prediction-initialized edited masks; unchanged regions retained as provided',independent_validation=False)
(out/'data_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n');print(json.dumps(audit,ensure_ascii=False,indent=2))
