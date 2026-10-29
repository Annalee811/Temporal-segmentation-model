# 3. Original temporal segmentation, V66

Each binary branch has its own original, independently trained25-channel
temporal classification head. The binary masks and binary probabilities are
fixed. This is not end-to-end retraining of the three segmentation backbones.
The saved100-epoch selections are U-Net37,DeepLabV3+14,SegFormer-B0 63.

```bash
python -m pip install -r requirements.txt
python ../tools/materialize_assets.py --stage 03_temporal_V66
python run.py infer --output ../work/V66
```

Features are recomputed from each adjacent RGB pair, frozen probabilities and
prediction-only root geometry. They include masked RGB, tip/junction proximity,
distance, orientation/coherence and local overlap cues. No temporal annotation
is passed to feature extraction or inference. Native256-pixel windows use128-pixel
stride, Hann blending and the original CPU BF16/FP32 arithmetic. The reference
environment reproduces all three models' four-period class maps pixel-for-pixel.

```bash
python run.py train --output ../work/V66_retraining
```

Optional training uses the released annotation-set-A target arrays, the original
paired manifest, random initialization, AdamW,100-epoch cosine scheduling,
150 unique windows plus50 focused samples per epoch, and class-weighted
CE/Dice/boundary/local/path losses. The maximum validation macroDice selects
the checkpoint; epoch0 is excluded. `training_curves.csv` contains only actual
numerical measurements. No conversation, launch record or per-operation history
is included. Full100-epoch retraining was not rerun for the snapshot checks.

Output codes are0=background,1=persistent,2=new,3=extension,4=T1baseline.
Evaluation reference set B is distinct from training set A. T1 is shown but
excluded from three-class temporal metrics. The common root-only evaluation
retains missed roots as FN; it does not evaluate background segmentation.

The binary arrays in this stage are the historical frozen inputs of V66.
They differ from the V35 binary experiment in stage2 and must not be replaced
when reproducing the original V66 scores.
