from pathlib import Path
import csv,hashlib,json,os
WORK=Path(os.environ['ROOT_REPRO_WORK']).resolve()
ROOT=WORK
OUT=WORK/'joint'
LAST=WORK/'local'
BASE=WORK/'base'
PERIODS=['T1','T2','T3','T4']
SEED=2026102813
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvout(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if not rows:return
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
