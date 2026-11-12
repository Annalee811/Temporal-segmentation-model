# Reproducible root-image matching and temporal analysis

This submission snapshot contains four computational stages. It contains the
core algorithms, frozen numerical inputs, model tensors, annotation arrays,
parameter sets, spatial splits and SHA-256 checksums needed to reproduce the
reported specimen results. Interactive annotation interfaces, conversation
records, operational logs, machine paths and unrelated experiments are excluded.

| Stage | Core implementation | Reproduction input/output |
|---|---|---|
| 1 | [Adjacent matching](01_adjacent_matching/README.md) | RoMa correspondences and frozen similarity reconstruction of four RGB observations |
| 2 | [Root/background segmentation](02_root_background/README.md) | V35 U-Net, DeepLabV3+ and SegFormer-B0 binary networks, exact test metrics and probabilities |
| 3 | [Temporal segmentation, V66](03_temporal_V66/README.md) | Original three 25-channel temporal heads on frozen binary support |
| 4 | [ConvLSTM/GAT refinement, V90](04_ConvLSTM_GAT_V90/README.md) | Learned graph relations, structural decoding and explicitly released reviewed-root supervision |

## Quick reproduction

Python 3.12, CPU PyTorch 2.8.0 and the pinned requirements are used for stages
1, 3 and 4 and the Torch branches of stage 2. The original DeepLabV3+ branch
uses Python 3.5/TensorFlow 1.12 in the supplied pinned CPU container; it is not
silently replaced by another architecture. GPU execution is optional and is
not the reference arithmetic. Allow 8 GB RAM for the individual Torch stages;
16 GB RAM and several GB of temporary disk space are recommended for the full
four-stage run. The upstream RoMa/DINOv2 weights require about 1.6 GB extra.

```bash
python -m pip install -r requirements.txt
python tools/verify_snapshot.py
python tools/materialize_assets.py
python 01_adjacent_matching/run.py prepare --raw-root 01_adjacent_matching/assets/raw --work work/registration
python 01_adjacent_matching/run.py render --work work/registration
python 02_root_background/infer.py --model unet --output work/binary/unet
python 02_root_background/infer.py --model segformer --output work/binary/segformer
python 03_temporal_V66/run.py infer --output work/V66
python 04_ConvLSTM_GAT_V90/run.py infer --output work/V90
python evaluate.py --v66 work/V66 --v90 work/V90/reviewed/full --output outputs/metrics
python figures.py --v66 work/V66 --v90 work/V90/reviewed/full --output outputs/figures
```


## What is and is not reproduced


## Binary segmentation reference, V35

| Model | Precision | Recall | Dice | IoU |
|---|---:|---:|---:|---:|
| U-Net | 0.9169 | 0.8437 | 0.8788 | 0.7838 |
| DeepLabV3+ | 0.7646 | 0.5492 | 0.6392 | 0.4698 |
| SegFormer-B0 | 0.8570 | 0.8155 | 0.8357 | 0.7178 |


## Evaluation protocol


| Output | Macro Dice | Macro IoU | Macro precision | Macro recall | Root accuracy |
|---|---:|---:|---:|---:|---:|
| V66 U-Net | 0.713237 | 0.626386 | 0.697819 | 0.736761 | 0.939259 |
| V66 DeepLabV3+ | 0.583733 | 0.468376 | 0.640777 | 0.536408 | 0.751044 |
| V66 SegFormer-B0 | 0.608901 | 0.486818 | 0.705272 | 0.539979 | 0.732661 |
| V90 reviewed U-Net refinement | 0.881774 | 0.723593 | 0.807313 | 0.820517 | 0.961004 |



## Verification, licensing and archiving

[CHECKSUMS.json](CHECKSUMS.json) freezes the code and resource shards.
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) records the actual verification scope.
[LICENSE](LICENSE) applies to original project code. Third-party source retains
its original notices under `vendor`; upstream pretrained weights retain upstream
terms. No blanket relicensing of third-party data or weights is asserted.

