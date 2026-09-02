# Baseline Specification V1

Status: frozen implementation specification and completed result record for the first experiment only.

Scope: Pure U-Net, Kvasir-SEG, reproducible training, and binary segmentation evaluation. This document does not specify ViL, VisionLSTM, xLSTM-UNet, Swin, or future comparison experiments.

The tensor convention follows `ARCHITECTURE_SPECIFICATION_V1.md`: tensors are `[B,C,H,W]`, images are RGB, masks are single-channel binary targets, and the model returns one-channel logits.

## 1. Dataset and split

### Decision

- **Source-backed:** Use the official [Kvasir-SEG dataset page](https://datasets.simula.no/kvasir-seg/) and cite Jha et al., *Kvasir-SEG: A Segmented Polyp Dataset*. The official page describes 1,000 paired polyp images and masks, separate image/mask folders with matching filenames, JPEG images, and variable source resolutions. It does not define a canonical project train/validation/test split.
- **Verified locally:** `data/Kvasir-SEG/images/` and `data/Kvasir-SEG/masks/` each contain 1,000 `.jpg` files. All filename stems are paired and all image/mask dimensions match. The auxiliary `kavsir_bboxes.json` is not used as the segmentation target.
- **Engineering choice — recommended:** Create a deterministic **70%/15%/15%** train/validation/test split from matched image-mask filename stems. Sort stems lexicographically, apply a seeded permutation with seed `42`, allocate the first 70% to training, the next 15% to validation, and the remainder to test. If all 1,000 pairs are present, this yields 700/150/150 pairs.
- **Alternative:** 80%/10%/10% using the same procedure. This provides more training data but fewer validation and test examples.

### Rationale

Splitting image and mask pairs together prevents correspondence errors. A frozen manifest makes the experiment reproducible and prevents later runs from silently changing the test set. The 70/15/15 choice gives the small dataset a larger held-out test set and validation set.

### Implementation record

The frozen manifest is `data/splits/kvasir_seg_seed42_70_15_15.json` with counts of 700 train, 150 validation, and 150 test pairs. The local dataset contains no verified subject/video grouping metadata used by this implementation, so the specified filename-stem split is the active baseline split. No external dataset download was performed by the implementation pass.

## 2. Input preprocessing and fixed resolution

### Decision

- **Engineering choice — recommended:** Resize every RGB image directly to `224 x 224` using bilinear interpolation.
- **Engineering choice — mask canonicalization:** Load each RGB JPEG mask, convert it to grayscale, threshold with `gray >= 128`, convert to binary `{0,1}`, then resize the binary mask to `224 x 224` using nearest-neighbor interpolation and represent it as float32 `[B,1,224,224]`. The value `128` is an engineering preprocessing choice based on the observed decoded ranges `0–8` and `246–255`; it is not an official Kvasir-SEG threshold.
- The model-output threshold remains probability `>= 0.5` and is distinct from the ground-truth mask preprocessing threshold.
- Convert image values from 8-bit `[0,255]` to floating-point `[0,1]`. Do not apply ImageNet mean/std normalization in the first from-scratch baseline.
- Use no stochastic augmentation in the first baseline. Training, validation, and test therefore share the same deterministic geometric preprocessing; validation and test additionally have no augmentation by definition.
- **Alternative:** Aspect-ratio-preserving resize followed by deterministic padding/cropping. This may reduce geometric distortion but introduces additional padding and crop conventions.

### Rationale

`224 x 224` is the starting resolution already identified in the experiment protocol and is compatible with the provisional Pure U-Net tensor ladder. Direct resize is the smallest reproducible preprocessing contract. Nearest-neighbor mask interpolation avoids creating fractional class labels. Scaling to `[0,1]` avoids introducing unverified dataset statistics.

### TBD

Whether aspect-ratio-preserving preprocessing improves the baseline is a future controlled preprocessing experiment, not part of this implementation.

## 3. Pure U-Net architecture

### Common tensor contract

Input: `[B,3,224,224]`.

Target: `[B,1,224,224]`, binary values.

Output: `[B,1,224,224]`, raw logits. Apply sigmoid only for loss/evaluation conversion; do not apply sigmoid inside the model head.

### Encoder and bottleneck

The following widths and resolutions are the baseline instantiation of the provisional ladder in `ARCHITECTURE_SPECIFICATION_V1.md`:

| Location | Tensor |
|---|---|
| Input | `[B,3,224,224]` |
| Encoder stage 0 | `[B,32,224,224]` |
| Encoder stage 1 | `[B,64,112,112]` |
| Encoder stage 2 | `[B,128,56,56]` |
| Encoder stage 3 | `[B,256,28,28]` |
| Bottleneck | `[B,256,14,14]` |

Each encoder stage and the bottleneck is a two-convolution block:

`Conv2d(3x3, padding=1, bias=False) -> InstanceNorm2d(affine=True, eps=1e-5) -> LeakyReLU(negative_slope=0.01)`, repeated twice.

- **Source-backed:** The source audit found this convolution/normalization/LeakyReLU pattern in the xLSTM-UNet CNN scaffold.
- **Engineering choice:** Reuse the pattern for the first Pure U-Net while keeping the model itself a conventional convolutional U-Net. The exact Pure U-Net width ladder is not established as a published configuration by the reference sources.
- **Alternative:** `Conv2d -> BatchNorm2d -> ReLU`. This is conventional but is more sensitive to small-batch behavior.

After each encoder stage except the last, apply `MaxPool2d(kernel_size=2, stride=2)`:

`224 -> 112 -> 56 -> 28 -> 14`.

- **Engineering choice — recommended:** Max pooling for all four downsampling transitions.
- **Alternative:** A learned stride-2 convolution. This adds parameters and makes the baseline less comparable to a conventional U-Net.

### Decoder and head

At each decoder level:

1. Bilinearly upsample by a factor of 2 with `align_corners=False`.
2. Apply a `1x1` convolution to match the corresponding skip width.
3. Concatenate with the encoder skip along channels.
4. Apply the same two-convolution block defined above.

Tensor flow:

| Decoder level | Upsampled/projected | Skip | Concatenated | Block output |
|---|---|---|---|---|
| 3 | `[B,256,28,28]` | `[B,256,28,28]` | `[B,512,28,28]` | `[B,256,28,28]` |
| 2 | `[B,128,56,56]` | `[B,128,56,56]` | `[B,256,56,56]` | `[B,128,56,56]` |
| 1 | `[B,64,112,112]` | `[B,64,112,112]` | `[B,128,112,112]` | `[B,64,112,112]` |
| 0 | `[B,32,224,224]` | `[B,32,224,224]` | `[B,64,224,224]` | `[B,32,224,224]` |

- **Engineering choice — recommended:** Bilinear upsampling plus a `1x1` channel projection.
- **Alternative:** Learned `2x2`, stride-2 transposed convolution. It is valid, but introduces additional learned upsampling parameters and a different decoder inductive bias.
- **Engineering choice:** Apply a final `1x1` convolution from 32 channels to 1 logit channel. No activation follows the head.

## 4. Loss

### Decision

- **Engineering choice — recommended:** Use an equally weighted combined loss:

  `L = 0.5 * BCEWithLogitsLoss + 0.5 * soft-Dice-loss`.

  The soft-Dice term uses sigmoid probabilities, sums over spatial pixels per image, averages over the batch, and uses `epsilon = 1e-6` in numerator and denominator:

  `soft_dice = (2 * sum(p*y) + epsilon) / (sum(p) + sum(y) + epsilon)`;

  `soft-Dice-loss = 1 - mean(soft_dice)`.

- **Alternative:** BCEWithLogitsLoss alone. It is simpler, but does not directly optimize an overlap measure.

### Rationale

BCE supplies pixelwise supervision while soft Dice directly addresses foreground overlap and reduces dependence on foreground/background pixel balance. The fixed weights and epsilon make the first run reproducible; they are not source-published settings for this project.

## 5. Optimizer, schedule, and run control

### Decision

- **Engineering choice — recommended optimizer:** AdamW with initial learning rate `1e-3` and weight decay `1e-4`.
- **Alternative:** Adam with learning rate `1e-3` and no decoupled weight decay.
- **Engineering choice — recommended schedule:** Cosine annealing over `100` epochs, with `T_max=100` and minimum learning rate `1e-6`; step the scheduler once after each completed training epoch.
- **Engineering choice — recommended batch size:** `4` images per update. This is a conservative initial choice for the documented low-memory local environment; no benchmark memory number is claimed.
- **Engineering choice — recommended run length:** maximum `100` epochs, with early stopping disabled for the first baseline. This fixes the training budget and avoids a second stopping hyperparameter.
- Save a `last` checkpoint after every epoch and replace a `best` checkpoint only when validation Dice strictly improves. Evaluate the `best` checkpoint on the test set. On an exact tie, keep the earlier checkpoint.

### Rationale

AdamW and cosine decay are a compact, widely available from-scratch training configuration. A fixed epoch budget plus validation-selected checkpoint separates model selection from final test evaluation and is easier to reproduce than an unspecified stopping rule.

### TBD

The batch size must be verified by the first implementation's memory check. If `4` does not fit, use the largest explicitly recorded lower batch size and treat that as a new run configuration; do not silently change it.

## 6. Randomness and reproducibility

### Decision

- **Engineering choice — recommended initial seed:** `42`.
- Seed Python, NumPy, the deep-learning framework, data shuffling, and any augmentation generator from this seed. Use deterministic backend settings where supported and record the framework/device versions.
- Record the split manifest, configuration, seed, best epoch, best validation Dice, and test metrics with each run.

### Rationale

The experiment protocol calls for one seed during early development and multiple seeds only for later final comparisons. One named seed keeps the first implementation auditable without implying statistical robustness from a single run.

### TBD

Bitwise determinism across different GPU, driver, and framework builds cannot be guaranteed before the execution environment is available. This is an environment limitation, not permission to change the protocol silently.

## 7. Evaluation protocol

### Decision

- Convert logits to probabilities with sigmoid and threshold at `0.5`. The threshold is fixed before test evaluation and is not tuned on the test set.
- Compute foreground metrics on each image's binary prediction and binary target, then report the macro mean across images. Also report standard deviation across test images.
- Report Dice, IoU, precision, and recall. Using `TP`, `FP`, and `FN` from the foreground pixels:

  - Dice: `2TP / (2TP + FP + FN)`
  - IoU: `TP / (TP + FP + FN)`
  - Precision: `TP / (TP + FP)`
  - Recall: `TP / (TP + FN)`

- **Engineering choice:** For a zero denominator, define a metric as `1` when both prediction and target satisfy the corresponding empty condition, and `0` otherwise. This convention must be implemented identically for validation and test.
- Use validation Dice for best-checkpoint selection. Do not use test metrics for training, checkpoint selection, threshold selection, or early stopping.
- Evaluate at the fixed `224 x 224` resolution with no post-processing or connected-component filtering in the first baseline.

### Rationale

Per-image macro averaging prevents larger images from dominating the summary and exposes variation across samples. A fixed threshold and no post-processing keep the comparison interpretable. The existing experiment protocol identifies Dice, IoU, precision, and recall as the required segmentation metrics; the threshold, aggregation, and edge-case rules are engineering choices.

### Alternative

Pixel-micro aggregation over all test pixels can be reported as a secondary analysis later, but it is not the primary baseline result.

## 8. Final classification of important decisions

| Decision | Classification |
|---|---|
| Kvasir-SEG official source and paired image/mask organization | SOURCE-BACKED |
| No canonical split established by the source documents | SOURCE-BACKED audit finding |
| 70/15/15 deterministic split, seed 42, frozen manifest | ENGINEERING CHOICE |
| 224 x 224 RGB input and binary mask contract | SOURCE-BACKED by project specification/protocol plus ENGINEERING CHOICE for fixed baseline resolution |
| Two 3x3 convolution blocks with InstanceNorm and LeakyReLU | SOURCE-BACKED pattern in reference scaffold; ENGINEERING CHOICE for Pure U-Net |
| Width ladder 32/64/128/256 and 14 x 14 bottleneck | ENGINEERING CHOICE aligned with architecture specification |
| MaxPool downsampling | ENGINEERING CHOICE |
| Bilinear upsampling plus 1x1 projection | ENGINEERING CHOICE |
| BCE/Dice combined loss and coefficients | ENGINEERING CHOICE |
| AdamW, `1e-3`, weight decay `1e-4` | ENGINEERING CHOICE |
| Cosine schedule, 100 epochs, batch size 4, best-validation-Dice checkpoint | ENGINEERING CHOICE |
| Initial seed 42 | ENGINEERING CHOICE aligned with protocol's single-seed development policy |
| Sigmoid threshold 0.5, per-image macro metrics, no post-processing | ENGINEERING CHOICE aligned with protocol metrics |
| Frozen local split manifest and verified filename-stem pairing | ENGINEERING CHOICE, verified in the local dataset |
| RGB JPEG mask grayscale conversion and `gray >= 128` binarization | ENGINEERING CHOICE based on observed separated decoded ranges; not an official threshold |
| Bitwise reproducibility across future hardware/software builds | TBD / environment-dependent |

## Recommended first baseline

Implement one from-scratch Pure U-Net with widths `32-64-128-256-256`, two `3x3` Conv–InstanceNorm–LeakyReLU operations per block, MaxPool downsampling, bilinear-plus-`1x1` decoder upsampling, and a one-logit `1x1` head. Train on the frozen 70/15/15 Kvasir-SEG split at `224 x 224`, using `[0,1]` inputs, the 50/50 BCE-plus-soft-Dice loss, AdamW at `1e-3`, cosine decay to `1e-6` over 100 epochs, batch size 4, and seed 42. Select the best validation-Dice checkpoint and report macro test Dice, IoU, precision, and recall at threshold 0.5.

This is the only architecture specified for implementation in this pass. ViL, Swin, and future augmentation or resolution studies remain outside scope.

## Completed baseline result and qualitative analysis

The approved Pure U-Net baseline was trained for the full 100-epoch budget in Colab using the frozen seed-42 Kvasir-SEG split. The best checkpoint was selected by validation Dice at epoch 40. The baseline remains a standardized lightweight research baseline, not a claim of faithful reproduction of the original 2015 U-Net.

### Training record

| Quantity | Recorded value |
|---|---:|
| Best validation Dice | `0.828427411334084` at epoch 40 |
| Validation IoU at best epoch | `0.7402202276` |
| Epoch-100 train loss | `0.0141890234` |
| Epoch-100 validation loss | `0.3737061760` |
| Epoch-100 validation Dice | `0.8146043172` |
| Epoch-100 validation IoU | `0.7249620276` |

Training loss continued to decrease after approximately epoch 40 while validation loss increased and validation Dice fluctuated or degraded. This is consistent with overfitting in this run and supports validation-Dice checkpoint selection.

### Test record

The following metrics were computed on the 150-image test split using the best epoch-40 checkpoint and the fixed prediction threshold `0.5`.

| Metric | Mean | Standard deviation |
|---|---:|---:|
| Dice | `0.8241105952570058` | `0.20469270150681784` |
| IoU | `0.7412125480073589` | `0.23600287350956678` |
| Precision | `0.8742652224627301` | `0.20029117152181355` |
| Recall | `0.8361236784857595` | `0.2207417757217905` |
| Test loss | `0.2446875516573588` | — |

The high per-image standard deviations indicate heterogeneous performance across the test images. These results are restricted to Kvasir-SEG and do not support broad medical generalization claims.

### Qualitative analysis record

Post-hoc visualization was completed with `scripts/visualize_predictions.py`. The deterministic fixed-sample grid included strong segmentations with Dice approximately `0.9424`, `0.9786`, `0.9546`, `0.8811`, and `0.9689`, as well as one moderate example near `0.725`.

The separate difficult-case grid contains the four lowest-Dice cases from the 150-image test set. They are not representative random samples. The observed failure modes were:

1. missed target with false-positive localization on another salient structure;
2. near-complete miss with foreground predicted in the wrong region;
3. severe false-positive over-segmentation;
4. under-segmentation or incomplete capture of a larger target.

The fixed-sample and lowest-Dice grids serve different post-hoc error-analysis purposes. Neither was used for training, checkpoint selection, hyperparameter selection, or any other experimental decision. No claim is made that a future ViL model will resolve these baseline failure modes.

The qualitative artifacts are written to the ignored experiment output directory:

`experiments/runs/baseline_pure_unet_seed42/visualizations/`

including the fixed-sample grid, lowest-Dice grid, and `visualization_metadata.json`.

## Next controlled experiment

The next experiment is **Architecture A: Pure U-Net + ViL/mLSTM bottleneck**. It will compare the frozen Pure U-Net against an independently trained model with the bottleneck feature-processing pattern adapted from the xLSTM-UNet implementation:

`[B,C,H,W] -> [B,H*W,C] -> ViL/mLSTM feature block -> [B,H*W,C] -> [B,C,H,W]`

The dataset, frozen split, preprocessing, evaluation metrics, and standardized training protocol should remain unchanged unless an explicit later decision records a change. This architecture is not implemented by this baseline record.
