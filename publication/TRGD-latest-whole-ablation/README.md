# TRGD整体流程消融：最新V35前景

- [整体消融PDF](TRGD_latest_whole_pipeline_ablation.pdf)
- [整体消融PNG](TRGD_latest_whole_pipeline_ablation.png)
- [整体消融TIFF，600dpi](TRGD_latest_whole_pipeline_ablation.tiff)
- [整体消融SVG](TRGD_latest_whole_pipeline_ablation.svg)
- [主图加消融，A–G合并PDF](V35_TRGD_with_whole_ablation.pdf)
- [主图加消融，A–G合并PNG](V35_TRGD_with_whole_ablation.png)
- [主图和消融双页PDF](V35_TRGD_main_and_ablation_2pages.pdf)

TRGD（V90）整体流程消融：使用最新V35第91轮根系前景、冻结V66时序先验和冻结V90权重。TRGD off绕过整套流程；TRGD on将ConvLSTM+GAT→Axis根轴约束→Junction交叉连接约束作为同一个整体开启。不拆分各组件，不重新训练。

18个观测位置、66张现有图像均完成前景一致性核验。准确度仅使用0353-2-B3已有时序标记的T2—T4，合计468203个参考根系像素；T1仅提供上下文。两组同样保留17438个V35漏检的参考根像素，不将漏检剔除后冒充主指标。背景不属于本次时序分类计分域。仅在450765个检出参考根像素上的条件指标另行保存。

TRGD off→on：新生根Dice83.01%→88.07%；既有根Dice95.28%→96.55%；Macro Dice89.14%→92.31%，提升3.17个百分点。新生根Precision82.01%→93.18%，Recall84.02%→83.50%。

本次为既有开发序列上、当前迁移部署接口的成套推理消融，使用已完成的配对预测结果重算；不代表重新训练的组件消融，也不是独立时序测试。接口与归档原前处理的实际差异见evaluation.json。V35根系/背景Dice仍为89.85%，与此处两类根时序Macro Dice分别报告。

标题和坐标使用黑色Times New Roman，PDF嵌入字体，三组件作为整体进行推理开关对照。所有模型权重、原始标记和既有结果保留。
