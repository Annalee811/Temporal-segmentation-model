# Pixel-level root annotation and dataset partitioning

## 1. Image sources and annotation units

Pixel-level reference masks for root–background segmentation were organized from temporally matched minirhizotron RGB images. The dataset comprised 244 existing annotated images of 256×256 pixels and 66 additional annotated images of 500×500 pixels, yielding 310 annotated parent-image files. The existing images originated from observation position 0353-2-B3 and represented 36 base spatial windows and 25 translated windows observed at four time points. The additional images covered 18 observation positions belonging to germplasms 0353, A017 and Q1, with 24, 22 and 20 images, respectively. The numbers of available additional images at T1–T4 were 18, 16, 15 and 17. Six missing observations were recorded as missing and were not replaced by blank images or all-background masks.

Additional windows were selected for discernible root structures. At each position, the same matched coordinates were used to define a 500×500 pixel window across all available time points. Image-file counts describe dataset organization rather than independent biological replication. Position, time point and source coordinates were retained for every parent image and derived patch.

## 2. Pixel-level annotation and mask encoding

Pixel-level annotation was performed using an in-house browser-based interface, and the final saved binary masks were used as training references. The interface supported foreground painting, background erasure and display zooming in the original image-coordinate system; display operations did not alter the dimensions or coordinates of exported masks. Following the root–soil segmentation objective used in root-image analysis [1], discernible root-tissue pixels were assigned to the foreground, whereas soil, particles, water-droplet highlights, cracks and other non-root structures were assigned to the background. Root colour, browning and temporal status did not define additional binary segmentation classes.

Masks were stored as lossless PNG files with 0 denoting background and 255 denoting root, and mapped to 0 and 1 for training. Only these two valid codes occurred in the final dataset. The submitted binary mask layer was decoded from run-length encoding at its original dimensions and paired with the corresponding RGB image using image identity, dimensions and SHA-256. Programmatic checks verified image–mask dimensions, admissible label values, file hashes and pixel correspondence after cropping. Colour, temporal and viability attribute layers were not used to train the V35 binary model. The final submitted binary layer defined each reference mask, and source information was retained with the annotation records.

## 3. Native 250×250 pixel patch generation

All parent images were converted to 250×250 pixel patches on the existing matched grid, without spatial rescaling or interpolation. Each 500×500 pixel additional image was partitioned into four non-overlapping quadrants with upper-left coordinates (0, 0), (250, 0), (0, 250) and (250, 250), producing 264 additional patches.

To preserve the borders of each existing 256×256 pixel image, four 250×250 pixel corner windows were extracted at offsets (0, 0), (6, 0), (0, 6) and (6, 6), producing 976 existing-data patches. Their union covered the complete 256×256 pixel parent image. The combined dataset therefore contained 1240 patches. A pixel-coverage array verified that every source pixel, including every annotated root pixel, occurred in at least one derived patch. The corner windows overlap extensively; 1240 is consequently the number of correlated training views, not the number of independent samples.

## 4. Spatial–temporal grouping and dataset partitioning

Partitioning was performed before patch allocation at the level of parent images and spatial association groups, rather than by independently randomizing 250 pixel patches. Consistent with structured-data partitioning principles [2, 3], windows at the same observation position were connected when their source-coordinate rectangles intersected. Connected components of this relation defined spatial groups. All dates, translated versions and derived patches belonging to the same connected group inherited one partition assignment. This constraint prevented direct reuse of the same source pixels across the current training and monitoring partitions; it did not constitute a whole-tube or whole-germplasm holdout.

Previous training, validation and test assignments were discarded for the current partition. With random seed 20261125, every group containing an additional image was assigned to training, together with any overlapping existing-data images. Forty-eight parent images were allocated to internal development monitoring from the remaining existing-only groups without splitting a group. Of 48 spatial groups, 41 were assigned to training and 7 to monitoring. Training contained 196 existing parent images and all 66 additional parent images, for 262 parents in total. Monitoring contained 48 existing parents. The resulting 1048 training patches and 192 monitoring patches represented 84.52% and 15.48% of the 1240 patches, respectively. No independent test set was established.

| Source | Parent files | 250px patches | Training | Monitoring |
|---|---|---|---|---|
| Existing | 244 | 976 | 784 | 192 |
| Additional | 66 | 264 | 264 | 0 |
| Total | 310 | 1240 | 1048 | 192 |

## 5. Training-time sampling and monitoring reconstruction

All 66 additional parent images and all 264 derived quadrants were used for training. Each epoch visited every one of the 1048 training patches once, without further random spatial subcropping. Over 100 epochs, this yielded 104800 patch visits, including 26400 visits to additional-data patches. The training-loss weight for an additional patch was set to six times that for an existing-data patch.

Augmentation was applied only after partitioning and only to training images. Geometric transformations consisted of random rotations by 0°, 90°, 180° or 270° and independent horizontal and vertical flips, each with probability 0.5. RGB images and masks received identical geometric transformations. Brightness multipliers and gamma exponents were independently sampled from 0.9–1.1 and applied only to RGB data. Monitoring images were not randomly augmented. For monitoring, probability maps were reconstructed on the parent-image grid using 250 pixel windows, a 188 pixel stride and Hann-weighted blending. Metrics were calculated at a fixed probability threshold of 0.5, counting each valid source pixel once rather than repeatedly scoring overlapping patches.

## 6. Scope of the development-monitoring results

Continued training inherited weights from previous experiments. An audit by parent-image identity showed that 44 of the 48 current monitoring images had previously been used to train inherited weights. The monitoring subset was therefore used only for internal development tracking and checkpoint selection. Its metrics were not interpreted as performance on strictly unseen images or as evidence of cross-tube or cross-germplasm generalization [4]. Non-overlapping groups in the current split do not remove historical data exposure embedded in inherited parameters.

The binary labels defined here were used for V35 root–background segmentation. Downstream TRGD (V90) used the same RGB input and fixed V35 foreground to assign newly emerged or existing-root labels only within root pixels; elongating and persistent states were merged into existing roots. These temporal outputs were distinct from the binary supervision defined above, and prediction images were not counted as additional reference annotations.

## References

[1] Smith AG, Petersen J, Selvan R, Rasmussen CR. Segmentation of roots in soil with U-Net. Plant Methods. 2020;16:13. doi:10.1186/s13007-020-0563-0. https://doi.org/10.1186/s13007-020-0563-0

[2] Roberts DR, Bahn V, Ciuti S, et al. Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography. 2017;40:913–929. doi:10.1111/ecog.02881. https://doi.org/10.1111/ecog.02881

[3] Valavi R, Elith J, Lahoz-Monfort JJ, Guillera-Arroita G. blockCV: An R package for generating spatially or environmentally separated folds for k-fold cross-validation of species distribution models. Methods in Ecology and Evolution. 2019;10:225–232. doi:10.1111/2041-210X.13107. https://doi.org/10.1111/2041-210X.13107

[4] Kapoor S, Narayanan A. Leakage and the reproducibility crisis in machine-learning-based science. Patterns. 2023;4:100804. doi:10.1016/j.patter.2023.100804. https://doi.org/10.1016/j.patter.2023.100804

