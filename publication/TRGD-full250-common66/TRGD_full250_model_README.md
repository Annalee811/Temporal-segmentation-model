# 更新后的组合模型：新V35 -> TRGD（V90）两类根系时序

模型包含更新后的V35根系/背景权重（310源图、1240个250px裁块、1048训练裁块（含补充264块）、完整100轮训练、验证选定第91轮）以及保留的V66/V90时序和图模型权重。现有V90迁移接口重新计算时序先验、ConvLSTM/GAT、Axis根轴约束和Junction交叉连接约束。最终foreground严格受新的V35支持域约束。根系/背景推理采用250px窗口、188px步长、0.5阈值，66张补充标记的264块全部用于本轮二分类训练。

## 对单个共有视野直接推理

安装runtime/requirements.txt。在一个目录放T1_common_RGB.png、T2_common_RGB.png等及common_domain.png；缺失的时期不放文件。也支持T1.png等文件名。

```bash
python runtime/predict_series.py --rgb-dir your_common_field --position your-position --output your_new_run
```

输出在your_new_run/predictions/your-position。只包含背景0、既有根1、新生根2；延伸与持续合并为既有。首个观测的可见根作为既有根参考基线。不会生成缺失期，跨缺期与上一个真实观测比较。所有RGB必须已经在同一共有视野和像素网格，本程序不做配准。

## 接入已保存的新V35完整运行结果

如果在本项目环境中已有含RGB、概率NPZ和root mask的work/V35_full250_common66：

```bash
python predict_latest_common.py --source /path/to/work/V35_full250_common66 --runtime runtime --output updated_temporal_run
```

也可在原仓库运行：

```bash
python 04_ConvLSTM_GAT_V90/run.py infer --common-source work/V35_full250_common66 --transfer-runtime work/V90_full250_common66/runtime --output work/V90_full250_common66
```

这次已实际推理18个位置、66张共有视野RGB，6个缺期在图中留空。逐像素检查确认V90没有增加或删除V35根系像素。预测结果由网络和结构约束生成；本轮没有重新训练时序权重，也没有把旧位置的准确率移用到当前图像。

## 方法来源

使用项目现有V90预训练迁移接口。骨架节点分区与邻接是重建接口；一个几何特征用最近交叉点度数代理；未包含原流程未发布的稠密局部修正预处理。因此不是原张量绑定预处理的完全数值复现，差异保留在preprocessing_provenance.json与模型配置中。原始旧权重保留在原项目中。
