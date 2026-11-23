"""Verify and materialize losslessly split resource archives from this snapshot."""
import argparse,hashlib,json,tempfile,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',nargs='+');a=p.parse_args()
 manifests=sorted((ROOT/'resources').glob('*/manifest.json'))
 if a.stage:manifests=[m for m in manifests if m.parent.name in a.stage]
 assert manifests,'No matching resource manifest; download all resource shards from the frozen snapshot.'
 for manifest in manifests:
  d=json.loads(manifest.read_text());stage=ROOT/d['stage'];stage.mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(prefix='root_materialize_') as temp:
   joined=Path(temp)/'assets.zip'
   with joined.open('wb') as out:
    for r in d['parts']:
     part=manifest.parent/r['file'];assert part.stat().st_size==r['bytes'] and sha(part)==r['sha256'],part.name
     with part.open('rb') as f:
      for b in iter(lambda:f.read(1024*1024),b''):out.write(b)
   assert sha(joined)==d['archive_sha256']
   with zipfile.ZipFile(joined) as z:
    for info in z.infolist():
     target=(stage/info.filename).resolve();assert target.is_relative_to(stage.resolve()),'Unsafe archive entry'
    assert z.testzip() is None;z.extractall(stage)
  for r in d['files']:
   assert sha(stage/r['file'])==r['sha256'],r['file']
  print('VERIFIED',d['stage'],len(d['files']),'resource files',flush=True)
if __name__=='__main__':main()
