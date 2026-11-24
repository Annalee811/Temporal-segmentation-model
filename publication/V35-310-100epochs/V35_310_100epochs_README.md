# V35：310 张合并数据重新划分、固定训练100轮

这次实验将原 V35 的全部244张图像与新增66张500×500像素标记合并为310张。旧训练/验证/测试划分全部废弃，重新划分为248张训练（196张旧图像、52张新增图像）和62张内部验证（48张旧图像、14张新增图像）。没有独立测试集，没有另外保留跨根管独立验证。

空间相交窗口连成同一组，包括全部日期、原图和平移裁剪，以及新增0353-2-B3的500px窗口与旧256px窗口的交集。48个空间组按种质覆盖进行80/20划分，不存在同一原始视野像素同时进入本次训练和验证的情况。文件 `data/manifest.csv` 包含全部310张图像的来源、坐标、哈希和新划分。新增标记使用上传JSON的原始二分类mask，不再保留上一版为旧验证/测试设置的额外ignore区域。

根系与背景是唯一目标：255=根系（包含新生、既有、基线等所有时序根），0=背景，128=原有不确定标记的ignore。JSON的temporal、color和viability层不参与训练。新增66张图像来自用户提交的预测初始化后修改文件，未修改区域按用户提交的状态保留；不能把所有标记称为从零独立绘制真值。

按用户明确选择，从原 V35 U-Net 最佳权重（原100轮实验第77轮）继续训练，保持ResNet18 U-Net结构。原权重完整保留，SHA256：2294f0002da52fc1636b01bed50e527015b597c0a7506c1eb7e38cd99635bd21。旧权重可能已经见过旧图像中现被重新划分为验证集的样本，所以本轮成绩仅为内部训练验证指标，不能解释为独立泛化能力。这里没有独立验证实验。

## 真实训练设置与结果

实际完成100轮，未启用早停。每轮248张训练图像全部访问一次，共24,800次图像访问、3,100次优化器更新。对500px图像随机裁剪原尺寸256px窗口，不缩放；使用旋转、翻转、亮度增强。AdamW，初始LR5e-5、weight decay1e-4，物理batch2、累积4、有效batch8。验证损失8轮不改善后降低学习率，最低1e-6；学习率调整不会停止训练。编码器BatchNorm运行统计沿用原V35，全部卷积及仿射参数可训练。损失为原V35 hard-negative BCE + Dice + 0.15 clDice。

每轮在全部62张验证图像的原分辨率有效像素上计分；500px图像用256px窗口、192px步长、Hann加权拼接；所有精度指标统一采用0.5阈值。验证损失使用相同V35组合目标，按滑窗patch平均。训练Dice来自增强后训练裁剪，验证Dice来自完整原尺寸图像，两条曲线的采样视野不同。

最佳验证权重在100轮全部完成后按验证Dice选定，第32轮；另外单独提供第100轮权重。训练曲线包含0轮初始化参考点及100个实际训练轮次，无插值造数。

| 内部验证指标 | 最佳验证权重（第32轮） | 第100轮权重 |
|---|---:|---:|
| Precision | 90.02% | 90.46% |
| Recall | 87.05% | 86.33% |
| Dice | 88.51% | 88.34% |
| IoU | 79.39% | 79.12% |
| Pixel accuracy | 99.15% | 99.14% |

背景像素占比大，Pixel accuracy不能单独代表根系分割质量，应同时查看Dice、IoU与Recall。

## 文件和推理

- `V35_310_100epochs_best_model.zip`：最佳权重、原V35备份、推理代码、训练代码、配置、清单与核心训练图。
- `V35_310_100epochs_epoch100_model.zip`：第100轮权重及相同推理接口。
- `V35_310_100epochs_dataset.zip`：全部310张RGB和二分类mask、分组划分清单、原上传JSON。
- `V35_310_100epochs_results.zip`：实际100轮记录、逐图指标、来源分层指标、Times New Roman黑字PDF/SVG/600dpi PNG训练曲线和示例。

```bash
python -m pip install -r requirements.txt
python infer_cross_tube.py --weights weights/V35_310_best.pt --image your_image.png --output predictions
# 第100轮模型：--weights weights/V35_310_epoch100.pt
```

输出与输入同尺寸的0/255根系mask及float32根系概率。不存在新生/既有分类输出。

## 复现

在本仓库内解压数据包，保持 `work/V35_310_100epochs/data/` 结构。把模型包中的 `V35_original_epoch77.pt` 放到 `work/V35_310_100epochs/weights/`；原发布权重也需位于 `02_root_background/assets/weights/unet/best_model.pt`，用于确认旧权重未改变（可从模型包原权重复制到该路径）。复制训练脚本至仓库的 `02_root_background/`。在没有现存训练状态的输出目录上运行：

```bash
python 02_root_background/train_merged_310.py --output work/V35_310_100epochs --epochs 100 --threads 4 --seed 20261124
```

断点恢复保留模型、优化器、学习率调度器、随机数状态和完整历史；本次实验协议必须保持相同。`protocol.json`、`selection.json`、`completion.json`、所有epoch_records及权重SHA256提供可核对的训练记录。
