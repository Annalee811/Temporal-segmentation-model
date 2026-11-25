# V35：补充66张、264个250px裁块全部用于训练

本轮实际完成100轮，无早停。从上一版V35_310_best.pt（前次100轮实验选定第32轮）初始化全部卷积/仿射权重，保留BatchNorm运行统计。原始第77轮及上一版第32轮、第100轮权重均完整保留，不以原第77轮作为本轮终点。

310张源图=旧244张256×256+补充66张500×500。500px完整四分为264个250px块；256px旧图使用偏移0和6的四角250px块共976个，保留边缘。总1240块，既不缩放，也不丢弃源像素。重叠子块不能视为独立样本。

**全部66张补充图像、264个子块均在训练集中，不留补充验证图。** 先强制补充图及与其空间相交的旧图组进入训练，再从剩余旧图的完整空间组中选48张父图验证。训练1048块（784旧+264补充），内部验证192块（全部旧数据），对应262张训练父图、48张旧验证父图。无独立测试。

每轮完整使用1048块一次，不再随机截小窗口。共104800次训练裁块访问，其中补充26400次。补充损失权重为旧样本6倍，占加权裁块暴露约66.89%；附加0.5Tversky项（假阴性权重0.7）。保留旧数据训练以维持既有识别能力。旋转、翻转、亮度和gamma增强不截去标记区域。

旧数据验证以父图原像素重建，250px滑窗、188px步长、Hann加权，原像素只计一次，阈值0.5。仅按旧数据内部验证Dice选权重，选中第91轮，另保留真实第100轮权重。补充图上的指标全部是训练拟合诊断，不用作验证或独立测试成绩。暖启动有历史来源接触，旧数据验证也不能解释为独立泛化。

## 真实结果

旧数据内部验证：上一版初始化Dice 89.27% → 本轮选定权重Dice 89.85%；选定权重Precision 91.34%，Recall 88.42%，IoU 81.57%。第100轮旧数据验证Dice 89.80%。

66张训练来源的完整500px前后预测与逐图拟合指标在supplemental_before_after文件中。模型输出由网络计算，不用提交标记覆盖预测。所有时序根类别统一视为根系，标记按上传状态保留。

training.pdf/svg/png使用真实100轮记录，黑字Times New Roman。(A)加权训练目标；(B)旧/补充训练Dice；(C)仅旧数据内部验证指标。examples.pdf/png对比同一RGB、提交标记、上一版模型和本轮模型。supplemental_predictions.zip含66张500px叠加PNG及根系mask，原尺寸不变。

## 文件和推理

V35_full250_all66_best_model.zip提供选定权重、模型与训练/推理代码、协议。三份旧权重分别放在backup_original77、backup_previous_best32、backup_previous_epoch100三个ZIP中，避免单个文件过大。epoch100_model.zip提供真实第100轮权重。dataset_old_A.zip、dataset_old_B.zip、dataset_new66.zip共1240个250px RGB/mask；dataset_parents.zip提供310张原父图及mask，便于重建验证与复现。

```bash
python -m pip install -r requirements.txt
python infer_full_250.py --weights weights/V35_full250_best.pt --image your_image.png --output predictions
# 第100轮：--weights weights/V35_full250_epoch100.pt
```

推理输出背景0/根系255和概率，不自动覆盖已发布V90。复现时四个数据包解压得到run/与source/，把模型包三份旧权重复制至run/weights/，代码放同一目录，运行：

```bash
python train_full_250.py --source source --output run --batch 8 --threads 4 --seed 20261125
```

来源SHA、裁剪位置、分组、每张源图完整覆盖证明、逐epoch使用数量、训练协议和模型SHA随包提供。


## 下载链接

- [V35_full250_all66_backup_original77.zip](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-full250-all66/V35_full250_all66_backup_original77.zip)
- [V35_full250_all66_backup_previous_best32.zip](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-full250-all66/V35_full250_all66_backup_previous_best32.zip)
- [V35_full250_all66_backup_previous_epoch100.zip](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-full250-all66/V35_full250_all66_backup_previous_epoch100.zip)
- [V35_full250_all66_best_model.zip](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-full250-all66/V35_full250_all66_best_model.zip)
- [V35_full250_all66_dataset_new66.zip](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-full250-all66/V35_full250_all66_dataset_new66.zip)
- [V35_full250_all66_dataset_old_A.zip](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-full250-all66/V35_full250_all66_dataset_old_A.zip)
- [V35_full250_all66_dataset_old_B.zip](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-full250-all66/V35_full250_all66_dataset_old_B.zip)
- [V35_full250_all66_dataset_parents.zip](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-full250-all66/V35_full250_all66_dataset_parents.zip)
- [V35_full250_all66_epoch100_model.zip](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-full250-all66/V35_full250_all66_epoch100_model.zip)
- [V35_full250_all66_examples.pdf](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-full250-all66/V35_full250_all66_examples.pdf)
- [V35_full250_all66_examples.png](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-full250-all66/V35_full250_all66_examples.png)
- [V35_full250_all66_results.zip](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-full250-all66/V35_full250_all66_results.zip)
- [V35_full250_all66_supplemental_predictions.zip](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-full250-all66/V35_full250_all66_supplemental_predictions.zip)
- [V35_full250_all66_training.pdf](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-full250-all66/V35_full250_all66_training.pdf)
- [V35_full250_all66_training.png](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-full250-all66/V35_full250_all66_training.png)
