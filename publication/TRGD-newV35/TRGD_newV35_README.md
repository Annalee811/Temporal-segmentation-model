# TRGD（V90）：使用最新版V35前景预测两类根系时序

V90的根系/背景输入已替换为310张标记、固定100轮训练后选出的新版V35（第32轮）概率与前景。重新计算V66时序先验、图结构以及V90 ConvLSTM/GAT、Axis根轴约束、Junction交叉连接约束；V66/V90时序权重保持冻结，没有使用新增根系/背景标记去虚构时序训练目标。

最终输出仅为0=背景、1=既有根、2=新生根。延伸与持续归为既有根。T1所有可见根作为既有根基线；后续类别比较此前真实观测，跨缺失期不补造中间影像。只处理18个共有视野、66张真实图像。PDF固定T1–T4四格，6个缺期留空。

所有时序输出的前景均与新版V35 mask逐像素相等，新增背景根系0像素，移除V35根系0像素。共有区域外没有标记，非根系RGB保持不变。该检查证明前景输入绑定一致，不代表时序类别的独立准确率。

本次沿用现有V90预训练迁移接口。它使用重建的骨架节点分区/邻接和一个最近交叉点度数几何代理，未包含原流程未发布的稠密局部修正预处理，因此不称为原张量绑定流程的完全数值复现，也不沿用旧位置的准确率作为本次成绩。原始V35与V90权重保留。

- [全部两类时序预测总览PDF（浏览版，缺期留空）](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_all_predictions_overview.pdf)
- [全部66张两类mask](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_all_two_class_masks.zip)
- [更新后的组合模型与运行代码](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_model_runtime.zip)

各位置ZIP包括共有RGB、两类mask、标记叠加PNG、V35前景、轴/轨迹推断编号与来源元数据。原尺寸PDF嵌入原像素PNG；浏览总览缩小显示以便查看，原始像素在各位置下载包中。轨迹编号是网络及约束推断的关联，不作为另一个根系语义类别。

| 位置 | 实际时期 | 原尺寸PNG和mask | 原尺寸PDF |
|---|---|---|---|
| 0353-1-B3 | T1,T2,T3,T4 | [ZIP (28.4 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_0353-1-B3.zip) | [PDF (19.4 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_0353-1-B3.pdf) |
| 0353-1-Z2 | T1,T2,T3,T4 | [ZIP (16.3 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_0353-1-Z2.zip) | [PDF (11.6 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_0353-1-Z2.pdf) |
| 0353-2-B3 | T1,T2,T3,T4 | [ZIP (37.3 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_0353-2-B3.zip) | [PDF (25.1 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_0353-2-B3.pdf) |
| 0353-2-LB2 | T1,T2,T3,T4 | [ZIP (56.5 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_0353-2-LB2.zip) | [PDF (39.2 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_0353-2-LB2.pdf) |
| 0353-2-LB3 | T1,T2,T3,T4 | [ZIP (58.1 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_0353-2-LB3.zip) | [PDF (40.3 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_0353-2-LB3.pdf) |
| 0353-3-LB2 | T1,T2,T3,T4 | [ZIP (24.2 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_0353-3-LB2.zip) | [PDF (17.4 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_0353-3-LB2.pdf) |
| 0353-3-LB3 | T1,T2,T3,T4 | [ZIP (47.3 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_0353-3-LB3.zip) | [PDF (33.6 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_0353-3-LB3.pdf) |
| A017-1-B2 | T1,T2,T4 | [ZIP (39.6 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_A017-1-B2.zip) | [PDF (26.9 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_A017-1-B2.pdf) |
| A017-1-Z2 | T1,T2,T4 | [ZIP (12.3 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_A017-1-Z2.zip) | [PDF (8.6 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_A017-1-Z2.pdf) |
| A017-2-2 | T1,T2,T3,T4 | [ZIP (38.5 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_A017-2-2.zip) | [PDF (26.7 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_A017-2-2.pdf) |
| A017-2-3 | T1,T2,T3,T4 | [ZIP (44.3 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_A017-2-3.zip) | [PDF (30.2 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_A017-2-3.pdf) |
| A017-3-B3 | T1,T2,T3,T4 | [ZIP (52.7 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_A017-3-B3.zip) | [PDF (35.4 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_A017-3-B3.pdf) |
| A017-3-Z2 | T1,T2,T3,T4 | [ZIP (55.7 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_A017-3-Z2.zip) | [PDF (38.4 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_A017-3-Z2.pdf) |
| Q1-1-B2 | T1,T3,T4 | [ZIP (14.3 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_Q1-1-B2.zip) | [PDF (8.9 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_Q1-1-B2.pdf) |
| Q1-1-Z3 | T1,T2,T3 | [ZIP (33.4 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_Q1-1-Z3.zip) | [PDF (23.9 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_Q1-1-Z3.pdf) |
| Q1-2-B3 | T1,T2,T3,T4 | [ZIP (60.4 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_Q1-2-B3.zip) | [PDF (39.6 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_Q1-2-B3.pdf) |
| Q1-2-Z2 | T1,T3,T4 | [ZIP (46.3 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_Q1-2-Z2.zip) | [PDF (31.9 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_Q1-2-Z2.pdf) |
| Q1-2-Z3 | T1,T2,T4 | [ZIP (54.4 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_Q1-2-Z3.zip) | [PDF (33.5 MB)](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/TRGD-newV35/TRGD_newV35_Q1-2-Z3.pdf) |

权重SHA256：
- weights/V90_joint_best.pt: `721e6c55e9522dfab54912bc70eee64e2460f364f7fbfba9c88e4996ad0d152a`
- upstream/03_temporal_V66/assets/weights/unet/best_temporal_head.pt: `794948f4a77075249678095546348d26fb359780669727643724ebca96a3001b`
- upstream/02_root_background/assets/weights/unet/best_model.pt: `43506267181c04ae44a0de280ce9c03839ea05f23468988b0be8e984cd52f940`
