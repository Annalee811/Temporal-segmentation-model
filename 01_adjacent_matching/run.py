"""Adjacent registration, fixed-geometry reconstruction and cumulative QC."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
import geometry as g
def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','render','confidence','all']);p.add_argument('--raw-root',type=Path);p.add_argument('--work',type=Path,required=True);p.add_argument('--parameters',type=Path,default=Path(__file__).with_name('registration_parameters.json'));a=p.parse_args();base=Path(__file__).resolve().parent;registry=json.loads(a.parameters.read_text());w=a.work.resolve();w.mkdir(parents=True,exist_ok=True)
    for stage in ['prepare','render','confidence'] if a.stage=='all' else [a.stage]:
        if stage=='prepare':
            assert a.raw_root is not None,'--raw-root is required';index={}
            for f in sorted(a.raw_root.rglob('*')):
                if f.is_file() and f.suffix.lower() in ['.jpg','.jpeg','.png','.tif','.tiff']:index.setdefault(g.sha(f),f)
            for key,e in registry.items():
                for t,h in e['input_sha256'].items():
                    assert h in index,('Required input checksum not found',key,t);src=index[h];dest=w/'inputs/raw'/key/(t+src.suffix.lower());dest.parent.mkdir(parents=True,exist_ok=True)
                    if dest.exists():assert g.sha(dest)==h
                    else:shutil.copy2(src,dest)
        elif stage=='render':
            expected=json.loads((base/'expected_outputs.json').read_text())
            for key,e in registry.items():
                for t,h in e['input_sha256'].items():
                    f=list((w/'inputs/raw'/key).glob(t+'.*'));assert len(f)==1 and g.sha(f[0])==h
                g.render_position(w,e,force=True)
                if key in expected:
                    for t,h in expected[key].items():assert g.sha(w/'deliverable'/key/'RGB'/(t+'_independent_stitched_RGB.png'))==h,('Output checksum mismatch',key,t)
        else:
            env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([str(base/'vendor'),env.get('PYTHONPATH','')]);subprocess.run([sys.executable,str(base/'infer_roma.py'),'--work',str(w),'--view','posed','--positions',*registry],env=env,check=True)
            area=[];scores=[]
            for key in registry:
                meta=json.loads((w/'deliverable'/key/'geometry_metadata.json').read_text());area.extend(meta['area_rows']);scores.extend(g.confidence_position(w,meta,force=True))
            (w/'tables').mkdir(exist_ok=True);g.csvwrite(w/'tables/common_area.csv',area);g.csvwrite(w/'tables/cumulative_confidence.csv',scores)
        print('COMPLETED',stage,flush=True)
if __name__=='__main__':main()
