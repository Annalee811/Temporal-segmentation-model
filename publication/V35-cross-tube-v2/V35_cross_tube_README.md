# V35 跨根管增强版：仅分割根系与背景

上传文件中的 mask 层作为唯一训练目标：255=根系、0=背景、128=忽略。新生、既有、持续、延伸及 T1 基线均视为同一根系类别；color、temporal、viability 层不用于训练。

原 V35 U-Net 第 77 轮权重完整保存在 weights/V35_original_epoch77.pt，SHA256：2294f0002da52fc1636b01bed50e527015b597c0a7506c1eb7e38cd99635bd21。架构仍为 ResNet18 U-Net，没有修改 V66/V90 的时序分类。

## 数据与真实训练

新增 18 个观测位置、8 个物理根管、3 个种质，共 66 张 500×500 px 原分辨率图像。上传掩膜中有 37 张相比初始化预测发生了二分类像素修改，修改像素共 56,345。其余区域按用户提供的文件保留，不能将其描述为全部从零独立绘制的真值。文件中 reviewed 标志均为 false，但用户明确将此文件指定为人工标记训练数据，因此未按界面标志删除数据，也未将其擅自标为已复核。

从原 V35 权重开始微调，全网络参数可训练，小批量情况下冻结编码器 BatchNorm 的运行统计；AdamW，初始学习率 5e-5，权重衰减 1e-4，batch=2、梯度累积=4，256×256 原生像素随机裁剪、旋转、翻转及亮度增强，损失沿用 V35 的 hard-negative BCE + Dice + 0.15 clDice。

分组开发训练使用 40 张新增图像及原 200 张训练图像；8 张 0353-3 根管图像用于验证；18 张 A017-3、Q1-1 根管图像用于测试。同一根管所有位置和日期保持在同一组。原 V35 验证20张、内部测试24张保持独立；新增 0353-2-B3 裁剪中与其重叠的像素已设为 ignore，训练不会监督这些像素。二分类数据中保留 ignore 并不代表新增分类。

开发训练最多20轮，至少6轮，验证分数5轮未提升则停止；分数为原验证 Dice 与跨根管验证 Dice 的均值。验证分数最高的轮次为第 1 轮。测试仅在选择完成后计分，不参与调参。500px 图像通过256px窗口、192px步长和Hann加权拼接预测；所有准确度指标在500px完整图像有效像素上以0.5阈值计算。

## 两种新版权重及指标适用范围

1. `development/best_model.pt`：分组开发版，用于留出跨根管测试集的公平对比，见结果包。它没有在18张跨根管测试图像上训练。
2. `weights/V35_cross_tube_all_labels.pt`：实际使用版，从原 V35 重新开始，以开发阶段选定的 1 轮训练预算，纳入全部66张新增图像和原200张训练图像。原验证/测试保护像素仍排除。所有新增根管已用于训练，因此该版本没有剩余的独立跨根管测试集，不能把开发版的跨根管指标冒充成该使用版的独立测试成绩。使用版仅在仍保留的原内部测试24张图像上另外计分。

留出跨根管18张图像（开发版 vs 原版）：
- Precision: 91.56% → 88.22%
- Recall: 66.28% → 69.01%
- Dice: 76.90% → 77.44%
- IoU: 62.46% → 63.18%
- Pixel accuracy: 99.09% → 99.08%

全部标记使用版在保留的原内部测试集：Dice=87.88%，IoU=78.38%，像素准确率=98.24%。根像素少，Accuracy 会被大量背景抬高，根系分割质量应结合Dice、IoU和Recall判断。

参考局限：留出根管的标签来自预测初始化后由用户提交的修改文件，指标表示模型与该文件的一致程度。这里只有两个测试根管，不能把像素数当作独立生物重复数，也不能据此宣称广泛跨环境泛化。

## 运行推理

需要 Python 3.12、torch 2.8 CPU、torchvision 0.23、transformers、numpy、Pillow。`python -m pip install -r requirements.txt`。使用版示例：

```bash
python infer_cross_tube.py --weights weights/V35_cross_tube_all_labels.pt --image your_image.png --output predictions
```

输出与输入同尺寸：`*_root_mask.png`（0=背景，255=根系）和 `*_root_probability.npy`。原版对照仅需替换为 `--weights weights/V35_original_epoch77.pt`。

## 复现训练

另下载训练数据包，按其中目录结构放入仓库 `02_root_background/assets/data` 及 `work/V35_cross_tube_v2/data`，将本模型包原权重置于 `work/V35_cross_tube_v2/weights/`。运行：

```bash
python 02_root_background/finetune_cross_tube.py --output work/V35_cross_tube_v2 --epochs 20 --threads 4 --seed 20261124
```

原始标记 JSON、来源哈希、数据划分、逐图指标、训练配置、两阶段完整曲线、选模记录与权重哈希一并保留。曲线均来自实际训练，无拟造数值。论文图使用 Times New Roman，黑色文字，提供 PDF/SVG 和600dpi PNG。
