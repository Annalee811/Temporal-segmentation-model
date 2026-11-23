"""Prepare the released paired split and run the original fixed100-epoch trainer."""
import argparse,shutil,subprocess,sys
from pathlib import Path
import common as c
HERE=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=c.MODELS,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--threads',type=int,default=2);a=p.parse_args()
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    if not (a.output/'data').exists():shutil.copytree(str(HERE/'assets/data'),str(a.output/'data'))
    protocol=c.read(HERE/'training_parameters.json');protocol['tensorflow_vendor_directory']=str(HERE/'vendor/deeplab')
    c.js(a.output/'protocol.json',protocol)
    script='train_tf.py' if a.model=='deeplab' else 'train_torch.py'
    subprocess.run([sys.executable,str(HERE/script),'--run',str(a.output),'--model',a.model,'--threads',str(a.threads)],check=True)
if __name__=='__main__':main()
