from pathlib import Path
import csv,json,hashlib,shutil,zipfile
P=Path('/workspace/shared/downloads');W=Path(__file__).resolve().parent;K='V35_updated_models_ABCD'
shutil.copy2(W/'preview.png',P/(K+'_preview.png'))
models={m:json.loads((W/(m+'_complete.json')).read_text()) for m in ['unet','deeplab','segformer']}
cn='''图：新版V35及保留基线的根系/背景分割表现。\n\nA：真实100轮损失曲线。U-Net为新版V35，使用1048个250×250训练裁块，选定第91轮权重；DeepLabV3+与SegFormer-B0采用原有基线权重，其A图曲线为历史训练记录，两种基线未重新训练。新版V35的加权训练目标与未加权监测目标不宜直接比较绝对大小。\n\nB：三种模型均在相同48张内部监测父图上重新计算像素级ROC，合计3145728个像素，其中根系153297个。所有模型均使用原像素250窗口、188步长和Hann概率拼接，阈值固定0.5。\n\nC：三种模型均对相同310张父图计算参考根长与预测根长：既有训练196张、补充训练66张、内部监测48张。根长定义为Zhang细化得到的骨架像素数，不作物理长度换算或剪枝。R²为1−SSE/SST；OLS R²为各组分别拟合线性关系的决定系数，图中实线仅用训练组拟合。残差为预测减参考。既有父图为256×256，补充父图为500×500；不将相关裁块数当成独立样本数。\n\nD：相同48张监测父图上的按参考类别行归一化混淆矩阵，显示比例和真实像素数量。每个父图网格计分一次，不重复计入其滑窗重叠部分。48张父图对应192个相关裁块；本轮未设置独立测试集。\n\n解释范围：当前48张监测父图中44张曾参与继承权重的前序训练，且模型的训练历史不同，因此图中结果用于当前开发监测，不作为统一从头训练的架构优劣或独立泛化结论。原始图像、参考标记及原模型权重均保留。\n'''
(P/(K+'_caption_CN.txt')).write_text(cn)
readme='''# 新版模型替换图：放大字体版\n\n沿用A–D排版；黑色Times New Roman。标题21pt、坐标轴标题16pt、刻度13pt、统计文字12.5pt、混淆矩阵百分比18pt。已增加间距并检查文本没有越出PDF页面。\n\n- [PDF矢量图](V35_updated_models_ABCD.pdf)\n- [PNG高清图，400dpi](V35_updated_models_ABCD.png)\n- [TIFF，600dpi，LZW无损压缩](V35_updated_models_ABCD.tiff)\n- [SVG可编辑矢量图](V35_updated_models_ABCD.svg)\n- [预览图](V35_updated_models_ABCD_preview.png)\n- [全部图件、数据和代码ZIP](V35_updated_models_ABCD_bundle.zip)\n- [中文图注](V35_updated_models_ABCD_caption_CN.txt)\n- [英文图注](V35_updated_models_ABCD_caption.txt)\n\n新版V35选定第91轮，完整训练100轮；DeepLabV3+及SegFormer-B0保留原权重。三者B/C/D均已在相同图像上重新评估，A图明确区分当前训练记录和历史基线曲线。\n\nROC及混淆矩阵统一使用48张监测父图；根长图统一使用262张训练父图和48张监测父图。本轮1048训练裁块、192监测裁块，没有额外独立测试集。原始标记及模型权重均不改写。\n\n当前监测父图存在前序训练暴露，基线训练历史也不同，不能将本图解释为严格独立测试或统一重新训练的架构比较。\n\n'''
for m,n in [('unet','新版V35 U-Net'),('deeplab','DeepLabV3+保留基线'),('segformer','SegFormer-B0保留基线')]:
 r=models[m];readme+=f"{n}：内部监测AUC {r['monitoring_auc']:.4f}，Dice {r['counts']['val']['Dice']*100:.2f}%，根长R² {r['regression']['val']['R2']:.4f}。\n\n"
(P/(K+'_README.md')).write_text(readme)
prov=json.loads((P/(K+'_provenance.json')).read_text());prov['font_sizes_pt']=dict(title=21,axis_label=16,tick=13,statistics=12.5,matrix_percentage=18,matrix_count=14);(P/(K+'_provenance.json')).write_text(json.dumps(prov,indent=2)+'\n')
with zipfile.ZipFile(P/(K+'_bundle.zip'),'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f in sorted(P.glob(K+'*')):
  if f.suffix!='.zip' and not f.name.endswith('_checksums.json'):z.write(f,f.name)
 for f in ['render_composite.py','evaluate_updated.py','deeplab_npz_forward.py','deeplab_equivalence.json','finalize_delivery.py']:z.write(W/f,'code/'+f)
checks={f.name:dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(P.glob(K+'*')) if not f.name.endswith('_checksums.json')}
(P/(K+'_checksums.json')).write_text(json.dumps(checks,indent=2)+'\n')
print('DELIVERY_READY',len(checks),'files')
