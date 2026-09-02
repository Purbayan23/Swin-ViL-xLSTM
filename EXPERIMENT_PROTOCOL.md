# Experimental Protocol

This remains the cross-model planning protocol. The first Pure U-Net implementation details are frozen in `BASELINE_SPECIFICATION_V1.md`.

## Data

- Primary dataset: Kvasir-SEG.
- Task: binary polyp segmentation.
- Dataset download: local copy present; no external download was performed by the implementation pass.
- Train/validation/test split: the first-baseline split is fixed in `data/splits/kvasir_seg_seed42_70_15_15.json`; later comparisons must reuse it.
- Dataset version/source: TBD.
- Image preprocessing: the first-baseline policy is direct `224x224` resize with bilinear interpolation; later comparisons must reuse it unless separately documented.
- Mask preprocessing: grayscale, engineering threshold `gray >= 128`, binary conversion, then nearest-neighbor resize. The threshold is not an official Kvasir-SEG threshold.
- Input tensor contract: RGB image tensor `[B,3,H,W]`.
- Target tensor contract: binary mask tensor `[B,1,H,W]`.
- Model output contract: segmentation logits `[B,1,H,W]`.

## Initial model sequence

1. Pure CNN U-Net.
2. CNN U-Net with a custom sequence-level ViL bottleneck.
3. CNN U-Net with ViL at selected deeper encoder stages.
4. Lightweight Swin comparison using the common experimental scaffold where possible.
5. Optional hierarchical ViL extension.

The first ViL experiment should use the custom sequence-level ViL integration pattern from xLSTM-UNet rather than treating the complete official VisionLSTM2 backbone as a direct Swin-block replacement.

## Common training variables

The following should be identical across controlled comparisons where technically possible:

- dataset and split;
- image resolution and preprocessing;
- augmentation;
- input normalization;
- optimizer;
- scheduler;
- loss;
- number of optimizer updates;
- early-stopping rule;
- checkpoint-selection rule;
- random seeds;
- prediction thresholding;
- evaluation implementation.

Values not yet established:

- For the first Pure U-Net baseline, optimizer, learning rate, scheduler, loss, epoch budget, batch size, early stopping, checkpoint selection, and seed are fixed in `BASELINE_SPECIFICATION_V1.md` and its JSON configuration.
- Future multi-model comparison details not covered by that baseline configuration remain TBD; one seed is acceptable during early pipeline validation, with three seeds preferred for a later final comparison.

## Metrics

Segmentation metrics:

- Dice;
- IoU;
- Precision;
- Recall.

Resource and scale measurements:

- trainable parameter count;
- total parameter count;
- FLOPs/MACs where meaningful;
- peak GPU memory;
- training time per epoch;
- total training time;
- inference time per image.

Boundary metrics such as HD95 may be added later, but are not required for the first baseline.

## Pretraining policy

The first controlled comparison should train all models from scratch. Pretrained Swin or ViL weights would otherwise introduce a major initialization confound relative to a randomly initialized CNN baseline.

Pretrained comparisons may be considered later as a separate experiment.

## Scientific rule

Do not claim that ViL is better, faster, or more efficient before measuring performance and resource use under the documented protocol.
