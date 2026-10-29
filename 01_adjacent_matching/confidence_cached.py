"""Recompute quality calculations from supplied original matching fields."""
import argparse,json,shutil
from pathlib import Path
import geometry
HERE=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args()
 for pair in (HERE/'assets/matching_fields').iterdir():
  if pair.is_dir():shutil.copytree(pair,a.work/'posed_roma_evidence/0353-2-B3'/pair.name,dirs_exist_ok=True)
 meta=json.loads((a.work/'deliverable/0353-2-B3/geometry_metadata.json').read_text());geometry.confidence_position(a.work,meta,force=True)
if __name__=='__main__':main()
