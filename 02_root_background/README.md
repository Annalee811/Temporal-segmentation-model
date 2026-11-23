# 2. Root/background segmentation, V35

This is the fixed100-epoch binary experiment corresponding to the following
internal test results, pooled over24 paired256-pixel views at threshold0.5.

| Model | Selected epoch | Precision | Recall | Dice | IoU |
|---|---:|---:|---:|---:|---:|
| U-Net | 77 | 0.9169 | 0.8437 | 0.8788 | 0.7838 |
| DeepLabV3+ | 63 | 0.7646 | 0.5492 | 0.6392 | 0.4698 |
| SegFormer-B0 | 87 | 0.8570 | 0.8155 | 0.8357 | 0.7178 |

The models are ResNet18 U-Net with native RGB skip, ResNet-v2-50 DeepLabV3+,
and SegFormer-B0 with native RGB residual refinement. Training masks use
0=background,255=root,128=ignore. Tensor-only selected and initial checkpoints,
all numerical100-epoch curves, data arrays and paired spatial split are supplied.

```bash
python -m pip install -r requirements.txt
python ../tools/materialize_assets.py --stage 02_root_background
python infer.py --model unet --evaluate --output ../work/V35/unet
python infer.py --model segformer --evaluate --output ../work/V35/segformer
python infer.py --model deeplab --evaluate --cached --output ../work/V35/deeplab_cached
python infer.py --model unet --output ../work/V35/full/unet
```

`--evaluate` runs real saved-model inference and checks exact test TP/FP/FN/TN.
`--cached` explicitly recomputes metrics from saved probabilities instead of
running a network. In preparation, fresh U-Net and SegFormer inference passed;
DeepLab was checked by cached probabilities only, because its legacy runtime
was unavailable. The last command performs qualitative full-image inference;
these full images are not the held-out test set.

DeepLab uses Python3.5/TensorFlow1.12, with fullFP32 original tensors and named
variable assignment. Docker and downloads of the pinned base image/wheel are
required for its CPU runtime; substituting a TF2 architecture is not equivalent.

```bash
python runtime/prepare_runtime.py --context runtime/build --build
sh tf_python.sh infer.py --model deeplab --evaluate --output work/V35/deeplab
```

Optional training starts from fresh random parameters, seed20261022, and runs
100epochs without early stopping. AdamW/TensorFlow Adam+L2 use initialLR3e-4,
physicalbatch2, accumulation4, effectivebatch8,25updates/epoch, weightdecay1e-4.
LR halves after8 nonimproving validation-loss epochs, minimum1e-6. The highest
validation Dice overepochs1-100 selects weights at fixedthreshold0.5.

```bash
python train.py --model unet --output ../work/V35_retraining
python train.py --model segformer --output ../work/V35_retraining
sh tf_python.sh train.py --model deeplab --output work/V35_retraining
```

There are200 training,20validation and24test views, based on25 spatial windows
plus translated training views. All dates at the same window retain their split;
all dates from this specimen belong to one outer development group. This is an
internal spatial test, not an independent tube test. Full inference uses256-pixel
windows,192-pixel stride and Hann blending.
