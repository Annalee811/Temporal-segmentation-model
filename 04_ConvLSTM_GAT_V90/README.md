# 4. ConvLSTM/GAT and structural refinement, V90

The learned core contains a2D ConvLSTM, two64-dimensional/four-head relation-gated
GAT layers, a root-state head and a learned junction-continuation head. The
binary foreground is immutable. The saved joint checkpoint was selected at
epoch78 after100 actual joint epochs, following100 pair-head pretraining epochs.

```bash
python -m pip install -r requirements.txt
python ../tools/materialize_assets.py --stage 04_ConvLSTM_GAT_V90
python run.py infer --output ../work/V90
```

The input is the released frozen graph/frame tensor dataset, node ownership,
geometry, initial local class priors, normalization and calibration parameters.
These are model inputs, not archived V90 output. This reproduction boundary is
explicit: the command does not claim to recreate every historical raw-to-graph
preparation experiment from RGB alone. Supervision arrays and split fields
support training; they are not substituted for neural predictions.

Inference recomputes node features, affinities and pair logits with the saved
weights, reconstructs root paths, applies cross-period axis-state constraints,
assigns a single owner at junctions and resolves mixed-axis classes. The
structural decoder uses the complete four-observation sequence, including later
observations; the complete workflow is offline, despite chronological neural
message passing. Root identity is distinct from physical contact or parentage.
An old mother and a newly observed lateral must remain separate biological roots.

The final `review_support.png` explicitly annotates one reviewed T2 root as new.
It affects3149 foreground pixels,2100 of which changed class relative to the
pre-review map. This human supervision is part of the released reviewed result,
not evidence that unchanged neural weights learned the correction. Inference
never reads the archived final V90 labels except after prediction for validation.

```bash
python run.py train --output ../work/V90_retraining
```

Optional training uses the released tensors, supervision and initialization.
It pretrains the continuation head then jointly trains ConvLSTM/GAT with BCE,
junction ranking, identity, root-state and chronological consistency objectives.
Full retraining was not repeated during snapshot preparation. The inference
check did rerun the learned forward pass and the complete structural/review
pipeline: neural arrays and all four final class maps matched exactly.

V90 is a learned-and-structured, reviewed workflow, not a pure architectural
ablation against V66. The source and evaluation annotations changed during
development. Pooled scores do not imply that every period improved.
