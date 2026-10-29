
"""Fetch a SHA-pinned official Intel TF1.12 wheel and optionally build its CPU image."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,urllib.request

WHEEL='intel_tensorflow-1.12.0-cp35-cp35m-manylinux1_x86_64.whl'
EXPECTED='026ddaf85cc6b6c83e8ba153f0ba9405f1249f92e877cf307528a63cdc9828d2'
BASE='tensorflow/tensorflow@sha256:1658b00f06cdf8316cd8a905391235dad4bf25a488f1ea989a98a9fe9ec0386e'
p=argparse.ArgumentParser();p.add_argument('--context',type=Path,default=Path('tf112_cpu_build'));p.add_argument('--build',action='store_true');a=p.parse_args();a.context.mkdir(parents=True,exist_ok=True)
with urllib.request.urlopen('https://pypi.org/pypi/intel-tensorflow/1.12.0/json',timeout=30) as r:
    info=json.load(r)
entry=next(v for v in info['urls'] if v['filename']==WHEEL)
assert entry['digests']['sha256']==EXPECTED
path=a.context/WHEEL
if not path.exists():
    with urllib.request.urlopen(entry['url'],timeout=60) as response,path.open('wb') as out:
        while True:
            part=response.read(1024*1024)
            if not part:break
            out.write(part)
h=hashlib.sha256()
with path.open('rb') as f:
    for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
assert h.hexdigest()==EXPECTED,'Downloaded wheel hash differs'
(a.context/'Dockerfile').write_text(f'''FROM {BASE}
COPY {WHEEL} /tmp/
RUN python -m pip install --no-index --disable-pip-version-check --no-deps --ignore-installed /tmp/{WHEEL} && rm /tmp/{WHEEL}
ENV OMP_NUM_THREADS=4 KMP_BLOCKTIME=0 TF_CPP_MIN_LOG_LEVEL=2
''')
print('Verified official wheel and wrote offline-install Dockerfile:',a.context)
if a.build:
    env=os.environ.copy();env['DOCKER_BUILDKIT']='0'
    subprocess.run(['docker','build','-t','root-tf112:local',str(a.context.resolve())],env=env,check=True)
