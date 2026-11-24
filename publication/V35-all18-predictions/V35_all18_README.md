# 新版 V35：全部18个观测位置的原始视野预测

模型已将原244张和新增66张二分类标记合并重新划分为248训练/62内部验证，并完整训练100轮。从原V35第77轮权重继续训练，内部验证选中第32轮权重，SHA256为 `43506267181c04ae44a0de280ce9c03839ea05f23468988b0be8e984cd52f940`。原V35权重保持不变。本轮不包含独立验证。

- [新版模型与原权重备份](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/c21a38de492fae861b3098ac09c3fd159da4bf9b/publication/V35-310-100epochs/V35_310_100epochs_best_model.zip)
- [实际100轮训练曲线PDF](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/c21a38de492fae861b3098ac09c3fd159da4bf9b/publication/V35-310-100epochs/V35_310_100epochs_training.pdf)
- [全部66张二值根系mask快速下载](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-all18-predictions/V35_all18_binary_masks.zip)

以下为18个位置、66张现有完整视野图像的真实网络预测。按位置分包，保留输入原始宽高，无缩放、形态学清理或人工后处理。未生成缺失时点图像。

每个包包含：
- `Tn_root_mask.png`：0=背景，255=根系，所有根系统一一类，阈值0.5。
- `Tn_root_probability.npz`：原分辨率float32概率，`numpy.load(path)['root_probability']`读取，未量化。
- `Tn_root_overlay.png`：根系标记叠加在原RGB上，黄色只表示根系。非根像素保持原图RGB不变。
- `Tn_metadata.json`：原图及结果SHA256、宽高、预测根系像素数、模型来源。

`root_area_fraction`为预测根系像素占总像素的比例，不是准确率，也不是生长速率。完整图像没有额外逐像素人工真值，因此此处只交付模型预测，不把它们称为已验证的准确分割。

| 种质 | 观测位置 | 时期 | 完整结果ZIP |
|---|---|---|---|
| 0353 | 0353-1-B3 | T1,T2,T3,T4 | [下载（47.7 MB）](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-all18-predictions/V35_all18_0353-1-B3.zip) |
| 0353 | 0353-2-B3 | T1,T2,T3,T4 | [下载（57.0 MB）](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-all18-predictions/V35_all18_0353-2-B3.zip) |
| 0353 | 0353-2-LB2 | T1,T2,T3,T4 | [下载（91.4 MB）](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-all18-predictions/V35_all18_0353-2-LB2.zip) |
| 0353 | 0353-2-LB3 | T1,T2,T3,T4 | [下载（93.8 MB）](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-all18-predictions/V35_all18_0353-2-LB3.zip) |
| 0353 | 0353-3-LB2 | T1,T2,T3,T4 | [下载（40.1 MB）](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-all18-predictions/V35_all18_0353-3-LB2.zip) |
| 0353 | 0353-3-LB3 | T1,T2,T3,T4 | [下载（79.7 MB）](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-all18-predictions/V35_all18_0353-3-LB3.zip) |
| A017 | A017-1-B2 | T1,T2,T4 | [下载（64.5 MB）](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-all18-predictions/V35_all18_A017-1-B2.zip) |
| A017 | A017-1-Z2 | T1,T2,T4 | [下载（19.6 MB）](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-all18-predictions/V35_all18_A017-1-Z2.zip) |
| A017 | A017-2-2 | T1,T2,T3,T4 | [下载（66.2 MB）](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-all18-predictions/V35_all18_A017-2-2.zip) |
| A017 | A017-2-3 | T1,T2,T3,T4 | [下载（77.6 MB）](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-all18-predictions/V35_all18_A017-2-3.zip) |
| A017 | A017-3-B3 | T1,T2,T3,T4 | [下载（91.6 MB）](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-all18-predictions/V35_all18_A017-3-B3.zip) |
| A017 | A017-3-Z2 | T1,T2,T3,T4 | [下载（96.0 MB）](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-all18-predictions/V35_all18_A017-3-Z2.zip) |
| Q1 | Q1-1-B2 | T1,T3,T4 | [下载（21.7 MB）](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-all18-predictions/V35_all18_Q1-1-B2.zip) |
| Q1 | Q1-1-Z2 | T1,T2,T3,T4 | [下载（38.7 MB）](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-all18-predictions/V35_all18_Q1-1-Z2.zip) |
| Q1 | Q1-1-Z3 | T1,T2,T3 | [下载（55.8 MB）](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-all18-predictions/V35_all18_Q1-1-Z3.zip) |
| Q1 | Q1-2-B3 | T1,T2,T3,T4 | [下载（93.6 MB）](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-all18-predictions/V35_all18_Q1-2-B3.zip) |
| Q1 | Q1-2-Z2 | T1,T3,T4 | [下载（70.2 MB）](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-all18-predictions/V35_all18_Q1-2-Z2.zip) |
| Q1 | Q1-2-Z3 | T1,T2,T4 | [下载（76.4 MB）](https://raw.githubusercontent.com/Annalee811/Temporal-segmentation-model/v66-publication-figures/publication/V35-all18-predictions/V35_all18_Q1-2-Z3.zip) |

缺失观测：A017-1-B2/T3、A017-1-Z2/T3、Q1-1-B2/T2、Q1-1-Z3/T4、Q1-2-Z2/T2、Q1-2-Z3/T3。

补充的66张500px图像叠加对照另在 V35-66-overlays 文件夹，分别显示上传根系标记与模型预测，二者不混为同一来源。
