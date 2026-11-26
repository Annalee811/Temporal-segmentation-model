from pathlib import Path
import json,hashlib,zipfile,shutil,subprocess,os
W=Path(__file__).resolve().parent;P=Path('/workspace/shared/downloads');K='TRGD_ABCD_uniform_fonts';stage=Path('/tmp/sci-annotation-methods-upload')
readme='''# A–D统一D图字号、图注与逐图分析\n\n图件保持所附复合图的A–C上排与D四时期图像阵列；正文、标题、数值、坐标、图例和比例尺统一为14pt黑色Times New Roman，A/B/C/D分图字母统一20pt。D最后一列统一命名TRGD（V90）。PDF嵌入字体，图像ROI原像素保留。\n\n- [图件PDF](TRGD_ABCD_uniform_fonts.pdf)\n- [图件PNG](TRGD_ABCD_uniform_fonts.png)\n- [TIFF，600dpi、Deflate无损压缩](TRGD_ABCD_uniform_fonts.tiff)\n- [SVG](TRGD_ABCD_uniform_fonts.svg)\n- [中英文图注及逐图分析Word](TRGD_ABCD_uniform_fonts_caption_analysis.docx)\n- [中文图注及分析](TRGD_ABCD_uniform_fonts_caption_analysis_CN.md)\n- [英文图注及分析](TRGD_ABCD_uniform_fonts_caption_analysis_EN.md)\n- [图D展示ROI的错分计数](TRGD_ABCD_uniform_fonts_D_ROI_error_counts.csv)\n- [全部文件ZIP](TRGD_ABCD_uniform_fonts_bundle.zip)\n\n保留用户图中的两套评估版本：A、D为归档V66 U-Net/TRGD比较（Macro Dice90.95%→93.59%），B、C为固定最新版V35前景的冻结权重整体流程消融（89.14%→92.31%）。图注明确区分，未改写原数值、原始标记或模型权重。所有时序准确度基于开发序列的参考根像素，不是独立时序测试。\n\n分析保留性能权衡和实际局限：C图新生根Recall84.02%→83.50%；最新消融的逐期T4新根Dice75.13%→69.46%；归档D图展示ROI的T3错分512→693。这些结果不支持“所有时期均改善”的表述。\n'''
(P/(K+'_README.md')).write_text(readme)
with zipfile.ZipFile(P/(K+'_bundle.zip'),'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f in sorted(P.glob(K+'*')):
  if not f.name.endswith(('_bundle.zip','_checksums.json')):z.write(f,f.name)
 for name in ['render_uniform.py','write_report.py','finalize_publish.py']:z.write(W/name,'code/'+name)
checks={f.name:dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in P.glob(K+'*') if not f.name.endswith('_checksums.json')};(P/(K+'_checksums.json')).write_text(json.dumps(checks,indent=2)+'\n')
url='https://github.com/Annalee811/Temporal-segmentation-model.git';branch='v66-publication-figures'
def run(args):return subprocess.check_output(args,cwd=stage,text=True).strip()
parent=run(['git','ls-remote',url,'refs/heads/'+branch]).split()[0]
try:run(['git','cat-file','-e',parent+'^{commit}'])
except subprocess.CalledProcessError:run(['git','fetch','origin',branch])
run(['git','read-tree',parent]);dest=stage/'publication/TRGD-ABCD-uniform-fonts';dest.mkdir(parents=True,exist_ok=True)
for f in P.glob(K+'*'):
 target=dest/f.name
 if target.exists():target.unlink()
 try:os.link(f,target)
 except OSError:shutil.copy2(f,target)
for name in ['render_uniform.py','write_report.py','finalize_publish.py']:shutil.copy2(W/name,dest/name)
(dest/'README.md').write_text(readme);run(['git','add','--','publication/TRGD-ABCD-uniform-fonts']);diff=run(['git','diff','--cached','--name-status',parent])
for line in diff.splitlines():
 status,path=line.split('\t',1);assert status in ['A','M'] and path.startswith('publication/TRGD-ABCD-uniform-fonts/'),line
print('FILES_TO_PUBLISH',len(diff.splitlines()),flush=True);tree=run(['git','write-tree']);commit=run(['git','-c','user.name=Codex','-c','user.email=codex@openai.com','commit-tree',tree,'-p',parent,'-m','Unify A-D figure text to D reference size and add bilingual caption with panel-specific measured interpretation'])
run(['git','push','origin',commit+':refs/heads/'+branch]);(W/'published_commit.txt').write_text(commit+'\n');print('PUBLISHED_COMMIT',commit,flush=True)
