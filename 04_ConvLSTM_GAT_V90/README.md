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


## 500px reference batch: preserve the selected V35 masks

The current entry uses the already registered66 input RGB images (500×500) and the exact masks in the user-selected `V35_full250_all66_supplemental_predictions.zip`. V35 is the sole root/background provider. V90 assigns existing/new labels only inside these fixed root pixels. No new registration, cropping, resizing or observation-field extraction is performed.

```bash
python 04_ConvLSTM_GAT_V90/predict_saved_v35.py --output work/V90_saved_V35_66
# Equivalent explicit entry:
python 04_ConvLSTM_GAT_V90/run.py infer --saved-v35-zip /workspace/shared/downloads/V35_full250_all66_supplemental_predictions.zip --output work/V90_saved_V35_66
```

Input RGB hashes and V35 mask ZIP members are bound in the manifest. Soft V35 probabilities are restored using the identical500px input and selected epoch91 checkpoint, and their0.5 threshold must exactly reproduce the selected masks. These cached probabilities supply V66 temporal features; the selected mask remains authoritative. Frozen V90 ConvLSTM/GAT, Axis and Junction are then evaluated on these inputs. Output fails if temporal foreground differs from the selected V35 mask.

Final classes are0=background,1=existing root,2=new root; extension and persistence merge into existing root. T1 visible roots form the existing reference baseline. Missing observations remain absent, with blanks in four-period figures. The position IDs from the selected66 inputs are retained unchanged.

The old full-field prediction remains archived as a different input-context run. All scripts and old weights are preserved. This current entry uses the existing pretrained V90 transfer interface: reconstructed skeleton partitions/neighbors, a nearest-junction degree proxy and no unreleased dense local-correction prepass. These departures are documented in provenance; no external temporal accuracy is claimed.

## Current complete native-image batch

Use the complete already matched18-position /66-image input set without registration, cropping or resizing. V35 is the sole root/background source on the same full RGB; V90 assigns temporal labels on its saved foreground. Complete-image V35/V90 inference at the unchanged selected epoch91 weights already exists; `reuse_matched_native.py` audits RGB hashes, mask hashes, checkpoint hashes and every foreground pixel before reusing it. The batch result is at `work/V90_all_matched_native`. Native PNG sizes and missing dates are preserved.

```bash
python 04_ConvLSTM_GAT_V90/reuse_matched_native.py
# Fresh temporal inference if the prepared V35 results or weights change:
python 04_ConvLSTM_GAT_V90/predict_latest_common.py --source work/V35_full250_common66 --runtime work/V90_full250_common66/runtime --output your_new_temporal_run
```

The source option takes already prepared V35 predictions and their exact RGB; it does not extract or align a field. A standalone500px reference input is a different neural context from a full native input and is not pasted or enlarged into these outputs. Within this full-input batch, V35 and V90 foregrounds must agree pixel-for-pixel.
