# Experimental Protocol

This remains the cross-model planning protocol. The first Pure U-Net implementation details are frozen in `BASELINE_SPECIFICATION_V1.md`.

The first Pure U-Net baseline is now complete. It used the frozen Kvasir-SEG split and approved 100-epoch protocol; the best validation-Dice checkpoint was epoch 40, with test Dice `0.8241105952570058` and test IoU `0.7412125480073589`. The complete result table and post-hoc qualitative analysis are recorded in `BASELINE_SPECIFICATION_V1.md`.

The corrected single-direction bottleneck comparison is **Architecture A0**. It completed the full 100-epoch run with the best validation-Dice checkpoint at epoch 51, validation Dice `0.8005566217`, test Dice `0.8175639115`, test IoU `0.7270158563`, precision `0.8651886918`, and recall `0.8180166952`. A0 is now frozen. The earlier incorrect-head, initialization-confounded epoch-78 run remains historical and incomplete.

The current experiment is **Architecture A1: Pure U-Net + alternating bidirectional ViL/mLSTM bottleneck**. A1 is an independent controlled ablation of frozen A0 introduced before Architecture B to test whether spatial sequence traversal affects the bottleneck result. It retains the same `[B,256,14,14] -> [B,196,256] -> [B,196,256] -> [B,256,14,14]` contract and all A0 controls, with no positional encoding or patch embedding. It follows the cited Vision-LSTM `ViLBlockPair`: an independent top-left-to-bottom-right block is followed by an independent bottom-right-to-top-left block, implemented by flipping the sequence before the second block and flipping its output back for spatial alignment. Outputs are composed sequentially, not averaged or concatenated.

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

1. Pure CNN U-Net (completed).
2. Architecture A0: single-direction ViL/mLSTM bottleneck (completed and frozen).
3. Architecture A1: alternating bidirectional ViL/mLSTM bottleneck ablation (current experiment).
4. Optional A2: A1 plus positional encoding.
5. Architecture B and later comparisons.

The A0 and A1 experiments use the custom sequence-level ViL integration pattern from xLSTM-UNet rather than treating the complete official VisionLSTM2 backbone as a direct Swin-block replacement. A1 adopts only the cited directional pair mechanism; positional encoding remains deliberately excluded.

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
