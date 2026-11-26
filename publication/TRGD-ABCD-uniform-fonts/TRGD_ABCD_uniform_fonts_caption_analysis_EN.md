# Figure caption

Figure X. Two-class temporal root classification, whole-pipeline ablation and qualitative comparison of Temporal Root Graph Decoding (TRGD, V90). (A) Precision, recall, Dice and intersection over union (IoU) for V66 U-Net and TRGD in the archived comparison. Gold and blue denote new and existing roots, respectively; dashed lines with open circles and solid lines with squares denote the two models. (B) New-root Dice, existing-root Dice and their unweighted mean (Macro Dice) with the complete TRGD process disabled or enabled on the latest fixed V35 foreground. (C) Corresponding changes in new-root precision, recall, Dice and IoU. Open circles and blue squares denote TRGD off and TRGD on; Δ is the difference in percentage points. (D) Original RGB, human reference, V66 U-Net and TRGD predictions for the same local field at T1–T4. Gold indicates new roots, blue indicates existing roots and white indicates the T1 root baseline. Scale bar: 100 pixels. ConvLSTM+GAT prediction, Axis root-axis constraints and Junction connection constraints are applied as one coupled TRGD process.

Evaluation scope. A and D retain the archived configuration, whereas B and C use the V35 epoch-91 foreground and the existing frozen TRGD transfer deployment interface. The comparisons therefore have different foreground configurations: Macro Dice 90.95%→93.59% and 89.14%→92.31% are reported separately. Quantitative temporal metrics use 468,203 reference-root pixels from T2–T4 of development sequence 0353-2-B3, with background excluded from the temporal scoring domain and T1 used only as context. Both conditions in B and C retain the same 17,438 reference-root pixels missed by V35. This figure does not provide an independent temporal test or repeated-training statistical inference.

# Panel-by-panel interpretation

## A. Archived model comparison

New-root precision increased from 86.71% to 94.51%, recall from 83.14% to 84.50%, Dice from 84.88% to 89.23% and IoU from 73.74% to 80.55%. The 4.34-percentage-point Dice gain was accompanied by a larger improvement in precision than in recall, consistent with fewer existing-root pixels receiving erroneous new-root labels within the reference-root domain. Existing-root Dice increased from 97.02% to 97.96%, and Macro Dice increased from 90.95% to 93.59% (+2.64 percentage points). These are pooled results for the archived comparison configuration.

## B. Whole-pipeline inference ablation

With RGB inputs, V35 foreground and model weights fixed, enabling the complete TRGD process increased new-root Dice from 83.01% to 88.07% (+5.07 percentage points), existing-root Dice from 95.28% to 96.55% (+1.27) and Macro Dice from 89.14% to 92.31% (+3.17). The larger new-root gain identifies temporal state discrimination as the principal improvement. Foreground differences were zero and both controls retained 17,438 missed reference-root pixels, so the gains arose from relabelling temporal states on fixed foreground rather than detecting additional roots. This bundled ablation estimates the joint effect of the process and does not isolate the contribution of each component.

## C. New-root precision–recall trade-off

Precision increased from 82.01% to 93.18% (+11.17 percentage points), whereas recall decreased from 84.02% to 83.50% (−0.53). Dice and IoU improved by 5.07 and 7.74 percentage points, respectively. Existing-root pixels erroneously classified as new roots decreased from 14,517 to 4,812, while correctly classified new-root pixels decreased slightly from 66,193 to 65,778. The output was therefore more conservative for new roots, reducing erroneous new-root labels at a small cost in detection. Because background is outside this temporal metric domain, these precision changes do not measure root–background segmentation performance.

## D. Spatial and temporal interpretation

The same native 420×320-pixel window is displayed across four observation times. White root overlays establish the T1 baseline; numerous gold new-root segments are visible at T2, while blue existing-root labels occupy much of the displayed network at T3 and T4. The adjacent reference and prediction columns allow spatial correspondence of root-state assignments to be inspected at crossings, fine branches and successive time points. Additional counts on the displayed reference-root pixels show errors decreasing from 1,701 to 130 at T2 and from 135 to 58 at T4, but increasing from 512 to 693 at T3, demonstrating spatial and temporal heterogeneity. This archived qualitative example supports interpretation of A; it neither replaces whole-field statistics nor demonstrates improved upstream binary foreground detection.

## Scope and limitations

The different foreground configurations in A/D and B/C prevent the 93.59% and 92.31% values from being interpreted as direct model deterioration or an additional gain. The pooled B/C improvements are not uniform across periods: the supplied per-period results show T4 new-root Dice decreasing from 75.13% to 69.46%. Conclusions are therefore restricted to pooled development-sequence performance, with a remaining limitation in late-period new-root classification. Correlated pixels within one sequence are not independent biological replicates, and no significance claim is made.
