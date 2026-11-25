# 4. Temporal Root Graph Decoding (TRGD, V90)

The default temporal segmentation has two root classes, matching V66.

| Pixel code | Class |
|---:|---|
| 1 | 既有根 / Existing root, including persistent roots and extensions |
| 2 | 新生根 / New root |

Code 0 is background. Code 4 marks T1 baseline root pixels. V90 uses graph
connections and chronology to refine the segmentation. Internal structural
decisions retain extension evidence, which is combined with persistent
evidence when the model emits temporal labels. The released graph training
already supervises root state with a binary new-root target. The frozen
pixel and local heads combine the two existing-root logits before returning
temporal class scores.

Run from the repository root.

```bash
python -m pip install -r 04_ConvLSTM_GAT_V90/requirements.txt
python tools/materialize_assets.py --stage 04_ConvLSTM_GAT_V90
python tools/materialize_assets.py --stage 03_temporal_V66
python 04_ConvLSTM_GAT_V90/run.py infer --output work/V90_merged
python evaluate.py --v90 work/V90_merged/prediction/full --output work/V90_merged/evaluation
```

`prediction/full` contains the final maps after automatic refinement of
root-axis continuity, temporal identity, and junction consistency. Evaluation
uses the same T2–T4 reference root pixels and two-class metrics as V66.

To retrain the graph and pair heads, use `run.py train` with a fresh output
directory. The optional `--legacy-three-class` flag exports the three
structural states.

## Root-segment proportions by tube

```bash
python 04_ConvLSTM_GAT_V90/root_segment_rates.py \
  --run 0353-2-B3=work/V90_optimized \
  --output work/V90_optimized/root_segment_rates
```

Repeat `--run POSITION=DIRECTORY` for each processed position. Each retained
topology segment takes its final axis label. A segment assigned to a joined
axis takes the destination axis label. Existing root covers both persistent
and extension segments. T1 baseline segments and topology bridges absorbed
into junctions are outside the denominator.

The script exports segment labels, counts and percentages by tube, and mean
percentages with equal weight per observed tube. For each tube, segments are
pooled across its positions within a period. The overall row pools all observed
periods within a tube before taking the mean across tubes. Missing periods
are omitted.

## Temporal baselines and joint pipeline ablation

Temporal Root Graph Decoding (TRGD) is the V90 inference pipeline:
ConvLSTM and GAT prediction, root-axis constraints, and junction constraints.
The ablation bypasses TRGD as a unit and evaluates the same run's frozen
upstream V66 maps. For TRGD off, the archived persistent and extension class
codes are merged in the upstream maps before scoring.

The frozen V66 input is both the temporal baseline and the TRGD-off
condition. The full V90 condition is V66 followed by TRGD. Both conditions
use identical U-Net foreground and the same T2–T4 reference root pixels.

```bash
python 04_ConvLSTM_GAT_V90/additional_evaluation.py development \
  --full work/V90_optimized --output work/V90_trgd_evaluation
```

For independent testing, set aside entire tubes before any training, tuning,
threshold selection or model selection. Supply their aligned two-class
predictions and separately annotated reference maps using a JSON manifest.
Each directory contains `T2_classes.png` and the equivalent filenames for
its other observed periods. Paths are relative to the manifest. For example:

```json
{
  "development_positions": ["0353-2-B3"],
  "test_cases": [
    {
      "position": "heldout-01-Z1",
      "periods": ["T2", "T4"],
      "reference": "heldout-01-Z1/reference",
      "prediction": "heldout-01-Z1/prediction"
    }
  ]
}
```

```bash
python 04_ConvLSTM_GAT_V90/additional_evaluation.py independent \
  --manifest heldout_manifest.json --output work/V90_independent
```

List every development position and replace the example case with real test
positions. The evaluator checks canonical tube overlap, reports scores by
position and tube, and pools the held-out confusion counts. The archive
contains the development run for 0353-2-B3; assessment on other tubes
requires their prepared predictions and annotations.


## Latest full250 V35 foreground for common-field temporal prediction

The current binary input is `V35_full250_best.pt`, selected at epoch91 after completing100 training epochs on1240 native250px crops from310 source images. All66 supplemental source images /264 quadrants were training data. The1048 training crops were visited in every epoch. Earlier V35 weights remain preserved. V66 epoch37 and V90 ConvLSTM/GAT epoch78 temporal/graph weights remain frozen.

```bash
python 02_root_background/predict_full250_common.py
python 04_ConvLSTM_GAT_V90/predict_latest_common.py
# Or explicitly bind the new common-field input to the versioned runtime:
python 04_ConvLSTM_GAT_V90/run.py infer --common-source work/V35_full250_common66 --transfer-runtime work/V90_full250_common66/runtime --output work/V90_full250_common66
```

The versioned runtime bundle supplies the selected V35 binary checkpoint,250px inference window and188px stride, and the frozen temporal checkpoints. Binary probabilities, V66 temporal priors, graphs, root-axis constraints and junction assignments are recomputed for all18 common fields /66 observed images. Pipeline configuration binds input hashes and all three model hashes. Export fails if any temporal foreground differs from the updated V35 mask.

Outputs contain only0=background,1=existing root and2=new root. Extension and persistence merge into existing root. T1 visible roots form the existing reference baseline. Missing observations stay absent from inference and blank in four-period figures; changes after a gap refer to the previous observed date. The original common RGB bytes, geometry and pixel grids remain unchanged.

This uses the existing V90 pretrained transfer interface. Reconstructed skeleton partition/neighbors and a nearest-junction degree cue differ from original tensor-bound preprocessing, and the unreleased dense local-correction prepass is absent. These departures remain in output provenance; no historical temporal score is assigned to these new predictions. The original tensor-bound reproduction remains available without `--common-source`.
