# 新版模型替换图：放大字体版

沿用A–D排版；黑色Times New Roman。标题21pt、坐标轴标题16pt、刻度13pt、统计文字12.5pt、混淆矩阵百分比18pt。已增加间距并检查文本没有越出PDF页面。

- [PDF矢量图](V35_updated_models_ABCD.pdf)
- [PNG高清图，400dpi](V35_updated_models_ABCD.png)
- [TIFF，600dpi，LZW无损压缩](V35_updated_models_ABCD.tiff)
- [SVG可编辑矢量图](V35_updated_models_ABCD.svg)
- [预览图](V35_updated_models_ABCD_preview.png)
- [全部图件、数据和代码ZIP](V35_updated_models_ABCD_bundle.zip)
- [中文图注](V35_updated_models_ABCD_caption_CN.txt)
- [英文图注](V35_updated_models_ABCD_caption.txt)

新版V35选定第91轮，完整训练100轮；DeepLabV3+及SegFormer-B0保留原权重。三者B/C/D均已在相同图像上重新评估，A图明确区分当前训练记录和历史基线曲线。

ROC及混淆矩阵统一使用48张监测父图；根长图统一使用262张训练父图和48张监测父图。本轮1048训练裁块、192监测裁块，没有额外独立测试集。原始标记及模型权重均不改写。

当前监测父图存在前序训练暴露，基线训练历史也不同，不能将本图解释为严格独立测试或统一重新训练的架构比较。

新版V35 U-Net：内部监测AUC 0.9935，Dice 89.85%，根长R² 0.9859。

DeepLabV3+保留基线：内部监测AUC 0.8353，Dice 60.76%，根长R² 0.5804。

SegFormer-B0保留基线：内部监测AUC 0.9629，Dice 74.09%，根长R² 0.9672。

