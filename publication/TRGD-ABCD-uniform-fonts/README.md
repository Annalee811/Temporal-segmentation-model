# A–D统一D图字号、图注与逐图分析

图件保持所附复合图的A–C上排与D四时期图像阵列；正文、标题、数值、坐标、图例和比例尺统一为14pt黑色Times New Roman，A/B/C/D分图字母统一20pt。D最后一列统一命名TRGD（V90）。PDF嵌入字体，图像ROI原像素保留。

- [图件PDF](TRGD_ABCD_uniform_fonts.pdf)
- [图件PNG](TRGD_ABCD_uniform_fonts.png)
- [TIFF，600dpi、Deflate无损压缩](TRGD_ABCD_uniform_fonts.tiff)
- [SVG](TRGD_ABCD_uniform_fonts.svg)
- [中英文图注及逐图分析Word](TRGD_ABCD_uniform_fonts_caption_analysis.docx)
- [中文图注及分析](TRGD_ABCD_uniform_fonts_caption_analysis_CN.md)
- [英文图注及分析](TRGD_ABCD_uniform_fonts_caption_analysis_EN.md)
- [图D展示ROI的错分计数](TRGD_ABCD_uniform_fonts_D_ROI_error_counts.csv)
- [全部文件ZIP](TRGD_ABCD_uniform_fonts_bundle.zip)

保留用户图中的两套评估版本：A、D为归档V66 U-Net/TRGD比较（Macro Dice90.95%→93.59%），B、C为固定最新版V35前景的冻结权重整体流程消融（89.14%→92.31%）。图注明确区分，未改写原数值、原始标记或模型权重。所有时序准确度基于开发序列的参考根像素，不是独立时序测试。

分析保留性能权衡和实际局限：C图新生根Recall84.02%→83.50%；最新消融的逐期T4新根Dice75.13%→69.46%；归档D图展示ROI的T3错分512→693。这些结果不支持“所有时期均改善”的表述。
