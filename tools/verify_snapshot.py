"""Check frozen project files, without reading credentials or external state."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 d=json.loads((ROOT/'CHECKSUMS.json').read_text())
 for r in d['files']:
  p=ROOT/r['file'];h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==r['sha256'],r['file']
 print('VERIFIED',len(d['files']),'frozen files')
if __name__=='__main__':main()
