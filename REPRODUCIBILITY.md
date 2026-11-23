# Reproduction checks

The reference preparation environment is Python3.12, CPU PyTorch2.8.0,
NumPy2.2.6, SciPy1.16.2 and scikit-image0.25.2 with CPU BF16 available.
The algorithm/data checks were performed before exporting the final snapshot.

| Component | Actually executed | Outcome |
|---|---|---|
| Frozen registration geometry | Reconstructed four RGB observations from raw input | Exact decoded RGB agreement with the downstream input arrays |
| Cached RoMa quality | Recomputed cumulative confidence from released original matching fields | All three adjacent-period quality calculations completed |
| RoMa dense matching | Not rerun in the preparation environment | Official implementation, pinned weights and original matching fields are supplied |
| V35 binary U-Net | Real checkpoint inference on24 internal test views | Exact TP100234,FP9086,FN18562,TN1444982; Dice0.8787985060 |
| V35 binary SegFormer-B0 | Real checkpoint inference on24 internal test views | Exact TP96883,FP16168,FN21913,TN1437900; Dice0.8357494382 |
| V35 TF1.12 DeepLabV3+ | Saved-probability metric recomputation; no fresh network inference | Exact TP65241,FP20084,FN53555,TN1433984; Dice0.6392384909; original FP32 tensors and runtime supplied |
| V66 | Fresh25-channel feature extraction and all three saved heads | Zero changed class pixels in all12 model/period outputs |
| ConvLSTM/GAT forward | Saved joint model recomputed from graph/frame tensors | Exact h,embedding,connection,normalized pair features and pair logits |
| V90 structural and reviewed output | Full path/junction decoder and explicit reviewed-root mask | Zero changed class pixels across all four observations |
| Model cleanup | Non-tensor checkpoint metadata removed | Tensor names,dtypes,shapes and bytes verified unchanged |

The archived final maps and neural arrays are comparison targets, not fallback
outputs. A mismatch is reported rather than silently overwritten. Different
hardware/library arithmetic may affect reproduction; the reference CPU settings
and supplied probability quantization are part of the specification.

Scientific numerical curves are preserved. Operational logs, conversation text,
launch states, session identifiers and machine-specific input paths are not.

The reported temporal scores are same-position development agreement under
revised annotation set B. They are not independent held-out/cross-tube accuracy.
Training used annotation set A; V90 additionally uses reviewed supervision and
offline multi-period structural decoding. Full training was not rerun for this
snapshot and its bitwise reproducibility across hardware is not asserted.

The released evaluation and plotting commands were also executed: temporal
metrics match the reference table, and training curves, four-period full/local
comparisons and confusion matrices were regenerated as PDF/TIFF.
