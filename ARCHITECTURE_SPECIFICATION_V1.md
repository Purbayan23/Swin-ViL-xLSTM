# Architecture Specification V1

Status: architecture specification only. No model implementation is included.

Date: 2026-08-24

This document converts the current project protocol and the inspected reference source trees into a tensor-level implementation target. It deliberately separates source-backed facts from engineering recommendations and unresolved decisions.

All files under `Reference/` remain immutable. The relevant source trees inspected were:

- `Reference/repositories/Swin-Unet-main/networks/swin_transformer_unet_skip_expand_decoder_sys.py`
- `Reference/repositories/Swin-Unet-main/networks/vision_transformer.py`
- `Reference/repositories/Swin-Unet-main/configs/swin_tiny_patch4_window7_224_lite.yaml`
- `Reference/repositories/vision-lstm-main/vision_lstm/vision_lstm2.py`
- `Reference/repositories/vision-lstm-main/vision_lstm/vision_lstm_util.py`
- `Reference/repositories/xLSTM-UNet-PyTorch-main/UxLSTM/nnunetv2/nets/UxLSTMBot_2d.py`
- `Reference/repositories/xLSTM-UNet-PyTorch-main/UxLSTM/nnunetv2/nets/UxLSTMEnc_2d.py`
- `Reference/repositories/xLSTM-UNet-PyTorch-main/UxLSTM/nnunetv2/nets/vision_lstm.py`

## Decision-classification key

- **DIRECT SOURCE SUPPORT**: directly visible in an inspected project document or source file.
- **STRONGLY SUPPORTED INFERENCE**: follows closely from source behavior, but is not itself a frozen project decision.
- **ENGINEERING CHOICE**: a proposed project design decision, not claimed as a paper or repository fact.
- **TBD**: cannot be resolved responsibly without a later project decision, compatibility test, or measurement.

# 1. Common segmentation contract

## 1.1 Input tensor

```text
X: [B, 3, H, W]
```

- `B`: batch size.
- `3`: RGB input channels.
- `H`: image height, measured in rows.
- `W`: image width, measured in columns.
- All image and feature tensors use channel-first NCHW layout unless explicitly converted to a sequence.

Classification: **DIRECT SOURCE SUPPORT** from `EXPERIMENT_PROTOCOL.md` for the RGB tensor contract; Kvasir-SEG is planned as an RGB image dataset.

## 1.2 Spatial resolution convention

The architecture must use one fixed spatial tensor size per experiment:

```text
H and W are fixed within a run.
H and W must be divisible by every planned downsampling factor.
```

The existing protocol recommends starting at `224x224`, but the final resolution is not frozen.

Classification:

- Fixed-size processing within a run: **STRONGLY SUPPORTED INFERENCE**.
- Starting at `224x224`: **ENGINEERING CHOICE** supported by the existing protocol.
- Final image resolution and aspect-ratio policy: **TBD**.

## 1.3 Target and output tensors

```text
Y:       [B, 1, H, W]   binary target mask
logits:  [B, 1, H, W]   raw foreground logits
```

The target mask represents background/foreground using values `{0, 1}`. The output is one foreground logit channel; a separate background logit is not required by this contract.

Classification:

- Shape and binary-mask contract: **DIRECT SOURCE SUPPORT** from `EXPERIMENT_PROTOCOL.md`.
- Target tensor dtype: **TBD**; it depends on the final loss choice.
- Sigmoid application: **STRONGLY SUPPORTED INFERENCE** that sigmoid should be applied for probability/thresholding outside the model, because the protocol specifies logits.

## 1.4 Sequence convention

When a spatial feature map is processed by a sequence-level ViL block:

```text
[B, C, H, W]
→ [B, H, W, C]
→ [B, H*W, C]
```

The sequence order must be row-major and invertible:

```text
token index = row * W + column
```

The inverse conversion must restore the same spatial location for every token.

Classification:

- NCHW to `[B,N,C]` conversion: **DIRECT SOURCE SUPPORT** from xLSTM-UNet `ViLLayer`.
- Row-major ordering: **DIRECT SOURCE SUPPORT** for the inspected flattening/rearrangement pattern; the project must still document the exact convention in implementation.

## 1.5 Skip-connection contract

Every encoder skip tensor must have the form:

```text
S_s: [B, C_s, H_s, W_s]
```

The corresponding decoder tensor must be restored to exactly `[B, C'_s, H_s, W_s]` before fusion. Any channel mismatch must be handled by an explicit projection. No implicit spatial interpolation should occur during skip fusion.

Recommended common rule:

```text
encoder stage
→ feature-processing block, if present
→ save processed skip tensor
→ next downsampling operation
```

This makes the CNN, ViL, and Swin variants expose comparable processed skip features.

Classification:

- Spatially aligned skip tensors: **STRONGLY SUPPORTED INFERENCE** from U-Net structure and the inspected xLSTM-UNet decoder.
- Saving the processed tensor after the stage processor: **ENGINEERING CHOICE**.
- Exact fusion operation and decoder channels: **TBD**.

# 2. Pure U-Net baseline

## 2.1 Reference status

No standalone project CNN U-Net implementation is present in `Reference/`. The xLSTM-UNet repository contains `UNetResEncoder` and `UNetResDecoder`, but these are nnU-Net-derived components with plan-driven widths, normalization, strides, and deep supervision. Reusing them would import additional framework behavior into the first baseline.

Therefore, the pure U-Net architecture below is a project design target, not a directly copied reference implementation.

Classification: **DIRECT SOURCE SUPPORT** that the pure U-Net must be the first model; the exact pure U-Net definition is **TBD**.

## 2.2 Provisional V1 tensor ladder

For the existing protocol's provisional `224x224` starting resolution, the following is a candidate lightweight ladder:

| Level | Tensor shape using candidate widths | Spatial size | Skip |
|---|---|---:|---|
| Input | `[B, 3, 224, 224]` | `224x224` | No |
| Encoder stage 0 | `[B, 32, 224, 224]` | `224x224` | Yes |
| Encoder stage 1 | `[B, 64, 112, 112]` | `112x112` | Yes |
| Encoder stage 2 | `[B, 128, 56, 56]` | `56x56` | Yes |
| Encoder stage 3 | `[B, 256, 28, 28]` | `28x28` | Yes |
| Bottleneck | `[B, 256, 14, 14]` | `14x14` | No separate decoder skip required |

The corresponding flattened sequence lengths would be:

```text
224x224 = 50176
112x112 = 12544
56x56   = 3136
28x28   = 784
14x14   = 196
```

This is a provisional target to make later tensor contracts concrete. It is not yet the final project configuration.

Classification:

- Four encoder stages plus one bottleneck: **ENGINEERING CHOICE**.
- Candidate widths `[32, 64, 128, 256, 256]`: **ENGINEERING CHOICE**.
- Resolution ladder at `224x224`: **ENGINEERING CHOICE** based on the existing protocol recommendation.
- Final number of stages, widths, and input resolution: **TBD**.

## 2.3 Encoder blocks

Candidate encoder stage:

```text
input feature
→ convolution block(s)
→ save skip tensor
→ 2x spatial downsampling, except before the bottleneck
```

The convolution block should preserve spatial resolution within a stage and change channels only where the stage definition requires it.

Potential source-aligned CNN choices include the residual blocks, InstanceNorm, and LeakyReLU used by the xLSTM-UNet factory. However, adopting those choices would make the baseline more nnU-Net-like rather than a neutral minimal U-Net.

Classification:

- Convolutional feature extraction: **DIRECT SOURCE SUPPORT** only at the broad U-Net level.
- Two `3x3` convolutions per stage: **ENGINEERING CHOICE**, not frozen.
- Residual versus plain convolution block: **TBD**.
- Normalization: **TBD**.
- Activation: **TBD**.
- Exact downsampling operator: **TBD**; candidates are max pooling or stride-2 convolution.

## 2.4 Bottleneck

The pure baseline bottleneck is a CNN feature block at the lowest spatial resolution. It must preserve its shape:

```text
[B, C_b, H_b, W_b] → [B, C_b, H_b, W_b]
```

For the provisional ladder:

```text
[B, 256, 14, 14] → [B, 256, 14, 14]
```

Classification:

- Shape-preserving bottleneck: **STRONGLY SUPPORTED INFERENCE**.
- Candidate `C_b=256`: **ENGINEERING CHOICE**.
- Number and type of bottleneck blocks: **TBD**.

## 2.5 Decoder

Candidate decoder contract:

```text
upsample to the next skip resolution
→ explicitly align channels
→ concatenate or otherwise fuse the matching skip
→ convolution block(s)
```

The decoder must reverse the encoder spatial ladder and produce `[B, C_0, H, W]` before the segmentation head.

Classification:

- U-Net-style mirrored decoder and skip fusion: **DIRECT SOURCE SUPPORT** at the architectural level.
- Nearest-neighbor plus `1x1` projection: **DIRECT SOURCE SUPPORT** for xLSTM-UNet, but not a project decision.
- Transposed convolution: **DIRECT SOURCE SUPPORT** as a Swin-Unet paper alternative, but not a project decision.
- Decoder upsampling method, block count, normalization, activation, and fusion order: **TBD**.

## 2.6 Segmentation head

Recommended final head:

```text
[B, C_0, H, W]
→ 1x1 convolution
→ [B, 1, H, W] logits
```

Classification:

- `1x1` class projection is directly used by Swin-Unet: **DIRECT SOURCE SUPPORT**.
- One foreground logit for the binary contract: **ENGINEERING CHOICE** supported by `EXPERIMENT_PROTOCOL.md`.
- Bias use and probability activation inside/outside the model: **TBD** except that the protocol output is logits.

# 3. ViL bottleneck

## 3.1 Selected ViL implementation

The first ViL bottleneck should use the custom sequence-level implementation pattern in:

```text
Reference/repositories/xLSTM-UNet-PyTorch-main/
UxLSTM/nnunetv2/nets/UxLSTMBot_2d.py
```

That wrapper calls the custom `ViLBlock` in:

```text
Reference/repositories/xLSTM-UNet-PyTorch-main/
UxLSTM/nnunetv2/nets/vision_lstm.py
```

It is not the complete official `VisionLSTM2` class.

Classification: **DIRECT SOURCE SUPPORT** and an established project decision.

## 3.2 Exact tensor transformation

For bottleneck feature `F_b`:

```text
F_b                    [B, C_b, H_b, W_b]
reshape/transpose      [B, H_b*W_b, C_b]
custom ViLBlock        [B, H_b*W_b, C_b]
transpose/reshape      [B, C_b, H_b, W_b]
```

The inspected xLSTM-UNet `ViLLayer` asserts that the channel dimension equals its configured `dim`. Therefore the first version should use:

```text
ViL dim = C_b
```

No projection is required in that case.

If a different internal ViL dimension is later desired:

```text
[B, C_b, H_b, W_b]
→ [B, H_b*W_b, C_b]
→ Linear(C_b, D)
→ ViL(D)
→ Linear(D, C_b)
→ [B, C_b, H_b, W_b]
```

Those projections are not part of the first recommended design.

Classification:

- Shape-preserving flatten/ViL/restore path: **DIRECT SOURCE SUPPORT**.
- `D=C_b` and no projections: **STRONGLY SUPPORTED INFERENCE** and recommended engineering choice.

## 3.3 Normalization and residual structure

The inspected custom `ViLBlock` contains:

```text
LayerNorm
→ custom ViLLayer
→ DropPath residual wrapper
```

The custom `ViLLayer` itself contains the mLSTM projections, causal 1D convolution, matrix-LSTM cell, learnable skip inside the mLSTM branch, gating, and output projection.

The xLSTM-UNet wrapper also defines a `self.norm`, but its forward method calls `self.vil` directly; the effective normalization for this path is the LayerNorm inside `ViLBlock`.

The first project adapter should not add another external residual or normalization layer around the custom block unless that becomes a documented ablation.

Classification: **DIRECT SOURCE SUPPORT** for the inspected custom block; external normalization/residual additions are **TBD** and should be avoided initially.

## 3.4 Sequence ordering

The xLSTM-UNet bottleneck wrapper uses:

```text
SequenceTraversal.ROWWISE_FROM_TOP_LEFT
```

It does not instantiate the official VisionLSTM2 block pair. The custom block therefore processes the flattened sequence in the top-left-to-bottom-right direction.

The official `VisionLSTM2` instead builds a `ViLBlockPair` containing top-left and bottom-right traversals. Using the full pair would be a different model choice.

Classification:

- Top-left sequence traversal for the first bottleneck: **DIRECT SOURCE SUPPORT**.
- Bidirectional/alternating pair: **DIRECT SOURCE SUPPORT** for official VisionLSTM2, but **not** the first bottleneck choice.

## 3.5 Positional encoding and patch embedding

The custom xLSTM-UNet bottleneck path does not call the official `VitPatchEmbed` or `VitPosEmbed2d` path. It receives an already spatially embedded CNN feature map and only changes its representation to a sequence.

For the first bottleneck:

- patch embedding inside ViL: avoid;
- additional VisionLSTM2 positional embedding: avoid;
- preserve the CNN feature-map coordinate order through the flatten/restore operation.

This avoids a second tokenization/downsampling mechanism and keeps the experiment focused on feature processing.

Classification: **STRONGLY SUPPORTED INFERENCE** from the xLSTM-UNet source and the established project decision.

## 3.6 Bottleneck contract

```text
Input:  [B, C_b, H_b, W_b]
Output: [B, C_b, H_b, W_b]
```

Required invariants:

- same batch size;
- same channel count;
- same spatial dimensions;
- one-to-one invertible token-to-pixel ordering;
- no pooling;
- no patch-size reduction;
- no decoder interface change.

# 4. Multi-stage ViL

## 4.1 Insertion rule

The recommended insertion point is:

```text
stage downsampling
→ CNN stage block
→ ViL sequence processing
→ save processed skip
→ next stage
```

This follows the order in the inspected xLSTM-UNet encoder: each CNN stage runs first, then the selected `xlstm_layers[s]` module runs, and the resulting tensor is appended to the skip list.

Applying ViL before downsampling is technically possible but has higher sequence length and is not the inspected xLSTM-UNet order.

Classification: **STRONGLY SUPPORTED INFERENCE** for the placement; it remains an **ENGINEERING CHOICE** for this project.

## 4.2 Candidate stage contracts

Using the provisional Tiny ladder from Section 2 and `224x224` input:

| Stage | Feature tensor | `H` | `W` | `C` | Sequence length `H*W` | Relative ViL memory pressure |
|---|---|---:|---:|---:|---:|---|
| Stage 0 | `[B, 32, 224, 224]` | 224 | 224 | 32 | 50176 | Very high; do not use first |
| Stage 1 | `[B, 64, 112, 112]` | 112 | 112 | 64 | 12544 | High |
| Stage 2 | `[B, 128, 56, 56]` | 56 | 56 | 128 | 3136 | Medium-high |
| Stage 3 | `[B, 256, 28, 28]` | 28 | 28 | 256 | 784 | Medium |
| Bottleneck | `[B, 256, 14, 14]` | 14 | 14 | 256 | 196 | Lowest |

The memory labels are qualitative consequences of sequence length and feature width. They are not benchmark measurements.

For every selected stage, the ViL contract is:

```text
[B, C_s, H_s, W_s]
→ [B, H_s*W_s, C_s]
→ ViL(C_s)
→ [B, H_s*W_s, C_s]
→ [B, C_s, H_s, W_s]
```

The returned feature is the skip tensor for that resolution.

## 4.3 Recommended first multi-stage configuration

After the bottleneck experiment is working, the first multi-stage candidate should process:

```text
Stage 2: [B, 128, 56, 56] → [B, 3136, 128] → ViL → [B, 128, 56, 56]
Bottleneck: [B, 256, 14, 14] → [B, 196, 256] → ViL → [B, 256, 14, 14]
```

Stage 3 at `28x28` may be substituted for Stage 2 if memory becomes limiting.

This recommendation is not a published architecture. It is an engineering choice that keeps ViL away from the two highest-resolution stages while testing both an intermediate and deepest scale.

The inspected xLSTM-UNet encoder uses a plan-dependent alternating condition:

```text
if bool(s % 2) XOR bool(n_stages % 2) and s > 1:
    use ViL
```

For a typical five-stage plan this selects stage indices 2 and 4. The exact plan-dependent stage set must not be assumed for the project model.

Classification:

- Plan-dependent selected-stage behavior: **DIRECT SOURCE SUPPORT**.
- Proposed Stage 2 plus bottleneck configuration: **ENGINEERING CHOICE**.
- Final multi-stage stage set: **TBD** until the common U-Net ladder and resource budget are frozen.

## 4.4 Official VisionLSTM2 at intermediate stages

If the complete official `VisionLSTM2` is ever evaluated at a U-Net stage, a wrapper would need to use approximately:

```text
input_shape=(C_s, H_s, W_s)
patch_size=1
pooling=None
mode="features"
```

This would produce a final feature sequence with one token per spatial location, but it would still introduce the VisionLSTM2 patch projection and positional embedding. `patch_size>1` would reduce the spatial token grid and create an unwanted second downsampling/tokenization mechanism for an intermediate feature map.

This is a secondary experiment, not the first ViL implementation.

# 5. Swin comparator

## 5.1 Reusable source components

The local Swin-Unet source provides reusable architectural components:

- `WindowAttention`;
- `SwinTransformerBlock`;
- `PatchMerging`;
- `PatchExpand`;
- `FinalPatchExpand_X4`;
- `BasicLayer`;
- `BasicLayer_up`;
- `PatchEmbed`.

The inspected lightweight configuration contains:

```text
input size:       224x224
patch size:       4
embed dim:        96
depths:           [2, 2, 2, 2]
decoder depths:   [2, 2, 2, 1] in YAML
attention heads:  [3, 6, 12, 24]
window size:      7
```

The source constructor actually builds decoder depths from reversed encoder `depths`, not from the exposed `depths_decoder` value. This must be treated as a source-specific implementation detail rather than silently copied into the project.

## 5.2 Native Swin-Unet tensor ladder

For `224x224`, patch size `4`, and embed dimension `96`:

```text
Patch embedding:  [B, 3136, 96]   = 56x56 tokens
Stage 0:          [B, 3136, 96]
Patch merging:    [B, 784, 192]    = 28x28 tokens
Stage 1:          [B, 784, 192]
Patch merging:    [B, 196, 384]    = 14x14 tokens
Stage 2:          [B, 196, 384]
Patch merging:    [B, 49, 768]     = 7x7 tokens
Stage 3/bottom:   [B, 49, 768]
```

The decoder reverses this hierarchy through patch expansion, skip concatenation, and linear projection, followed by final 4x expansion and a `1x1` segmentation head.

These native Swin-Unet resolutions are not identical to the provisional pure CNN U-Net ladder in Section 2.

## 5.3 Common-framework integration

For a fair common-scaffold comparison, the Swin feature processor should expose the same external contract as the CNN and ViL feature processor:

```text
[B, C_s, H_s, W_s]
→ internal Swin token processing
→ [B, C'_s, H_s, W_s]
```

If `C'_s` differs from the common decoder channel count, use an explicit `1x1` projection.

Two possible comparison modes exist:

1. **Reference-faithful Swin-Unet**: retain the native patch-4, 56/28/14/7 hierarchy and original decoder. This is a valid external baseline but not a tightly controlled replacement experiment.
2. **Common-scaffold Swin**: reuse Swin blocks or layers at the same externally defined stage resolutions as the CNN and ViL models. This is the preferred controlled comparison, but it requires adapters and is not the original Swin-Unet unchanged.

The project must not describe a Swin block and a ViL block as one-to-one equivalents. They differ in spatial organization, positional treatment, local/global processing, normalization, and computational behavior.

Classification:

- Swin component behavior: **DIRECT SOURCE SUPPORT**.
- Common-scaffold adapters: **ENGINEERING CHOICE**.
- Exact Swin comparator configuration: **TBD**.

# 6. Model-scale options

The following are provisional project scale envelopes, not official named model variants. Parameter counts for the complete project models are TBD because the pure U-Net blocks, decoder, projections, and final ViL placement have not been implemented or measured.

All rows use the provisional spatial ladder in Section 2.

| Scale | Candidate widths `[stage0, stage1, stage2, stage3, bottleneck]` | Candidate ViL depth | Sequence lengths available | Expected pressure |
|---|---|---:|---|---|
| Tiny | `[32, 64, 128, 256, 256]` | 2 block pairs at selected locations | `50176, 12544, 3136, 784, 196` | Appropriate first target; avoid high-resolution ViL |
| Small | `[48, 96, 192, 384, 384]` | 4 block pairs at selected locations | Same spatial lengths | Moderate; likely use only deep stages |
| Medium | `[64, 128, 256, 512, 512]` | 6 block pairs at selected locations | Same spatial lengths | High; defer until the pipeline is stable |

The candidate widths and depths are **ENGINEERING CHOICES**, not source-derived model definitions. The word `depth` for ViL refers to the official VisionLSTM2 convention of block pairs only if that implementation is used; the custom xLSTM-UNet wrapper itself inserts one custom `ViLBlock` at each selected location.

## 6.1 Source-derived scale references

The official Vision-LSTM paper reports the following full image-backbone reference scales at its classification setting:

| Source model | Latent dimension | Depth convention | Reported parameter/FLOP scale |
|---|---:|---:|---|
| ViL-T | 192 | 12 block pairs | approximately 6M parameters, 1.5G FLOPs |
| ViL-S | 384 | 12 block pairs | approximately 23M parameters, 5.1G FLOPs |
| ViL-B | 768 | 12 block pairs | approximately 89M parameters, 18.6G FLOPs |

These values describe the published full Vision-LSTM image backbone, not a bottleneck-only segmentation module. They must not be used as exact parameter counts for the proposed U-Net variants.

The Swin-Unet source provides the lightweight configuration dimensions and depths in Section 5, but no complete project parameter count has been calculated here.

## 6.2 Measurements required later

For each complete implemented model, measure rather than infer:

- trainable and total parameter count;
- FLOPs/MACs where the measurement is meaningful for the implementation;
- peak training memory;
- peak inference memory;
- time per epoch;
- inference time per image.

No benchmark memory number is asserted in this specification.

# 7. Fair comparison

## 7.1 Components to keep identical

Across the Pure U-Net, ViL, and Swin models, keep these identical where technically possible:

- RGB input contract;
- image and mask resolution;
- image interpolation and normalization;
- mask nearest-neighbor resizing;
- train/validation/test split;
- augmentation;
- output shape `[B,1,H,W]`;
- decoder;
- segmentation head;
- loss;
- optimizer;
- scheduler;
- optimizer-update budget;
- checkpoint-selection rule;
- random seeds;
- thresholding;
- metric implementations and aggregation.

The first controlled comparison should use random initialization for all models, as already established in the project protocol.

## 7.2 Components that may be shared structurally

The strongest controlled design is:

```text
same input contract
same encoder resolutions
same skip tensors
same decoder
same segmentation head
different feature-processing module
```

The feature processor can be:

- CNN block for the Pure U-Net;
- custom sequence-level ViL block for the ViL model;
- Swin block/layer for the Swin model.

This common-scaffold model is not identical to the original published Swin-Unet or xLSTM-UNet. It is a controlled project comparison inspired by them.

## 7.3 Unavoidable differences

The following cannot be made exactly identical:

- Swin local window attention versus ViL sequence processing versus CNN locality;
- Swin relative-position bias versus ViL sequence order/positional mechanisms;
- LayerNorm and mLSTM internal operations versus CNN normalization;
- different block-depth semantics;
- different parameter-to-activation relationships;
- different kernel and hardware implementations;
- different native tokenization assumptions;
- possible ViL float32 handling in the inspected xLSTM-UNet wrapper when receiving float16 input;
- pretrained-weight compatibility, which is excluded from the first comparison.

These differences should be disclosed rather than described as exact architectural equivalence.

# 8. Recommended first implementation

## Recommendation

Implement exactly one architecture first:

> **Pure CNN U-Net, Tiny provisional scale, with four encoder stages, a shape-preserving CNN bottleneck, mirrored decoder, and one-channel logit head.**

Provisional tensor ladder:

```text
Input:          [B, 3, 224, 224]
Stage 0:        [B, 32, 224, 224]   skip
Stage 1:        [B, 64, 112, 112]   skip
Stage 2:        [B, 128, 56, 56]    skip
Stage 3:        [B, 256, 28, 28]    skip
Bottleneck:     [B, 256, 14, 14]
Decoder:        restore 28, 56, 112, 224 resolutions with matching skips
Output:         [B, 1, 224, 224] logits
```

This recommendation is:

- technically valid;
- the easiest model to debug;
- small relative to the source paper-scale models;
- independent of ViL and Swin-specific compatibility issues;
- required for validating the data, loss, metrics, checkpointing, and evaluation pipeline.

The following details remain open even for this first model:

- exact convolution block definition;
- normalization;
- activation;
- downsampling operator;
- decoder upsampling operator;
- exact width cap;
- exact input resize/aspect-ratio policy;
- loss and optimizer.

Therefore this is the recommended **architecture target**, not authorization to implement before those TBD items are resolved.

Classification: **DIRECT SOURCE SUPPORT** for the first-model ordering; the concrete Tiny ladder is an **ENGINEERING CHOICE**; unresolved layer details are **TBD**.

# 9. Implementation dependencies

## 9.1 Source modules that may eventually be adapted

Potential future adaptations, without modifying the originals:

- `xLSTM-UNet-PyTorch-main/UxLSTM/nnunetv2/nets/UxLSTMBot_2d.py` for the NCHW-to-sequence bottleneck wrapper;
- `xLSTM-UNet-PyTorch-main/UxLSTM/nnunetv2/nets/vision_lstm.py` for the custom `ViLBlock`, `MatrixLSTMCell`, `LayerNorm`, and traversal implementation;
- `xLSTM-UNet-PyTorch-main/UxLSTM/nnunetv2/nets/UxLSTMEnc_2d.py` for plan-dependent multi-stage placement logic;
- `vision-lstm-main/vision_lstm/vision_lstm2.py` only for a later complete-backbone or feature-mode ablation;
- `Swin-Unet-main/networks/swin_transformer_unet_skip_expand_decoder_sys.py` for Swin blocks, merging, and expansion;
- `Swin-Unet-main/configs/swin_tiny_patch4_window7_224_lite.yaml` as a reference configuration, not a project configuration.

## 9.2 Source modules that should remain untouched

- all files under `Reference/`;
- all three raw repository trees;
- `Reference/notebooks/cifar10.ipynb`;
- original repository training scripts and entry points;
- original repository configuration files;
- original paper PDFs and existing audit/protocol files.

Future project code must copy or deliberately adapt behavior into project-owned files rather than editing reference code.

## 9.3 External dependencies expected later

The reference source trees indicate future implementation will require some combination of:

- PyTorch;
- torchvision;
- einops;
- NumPy and Pillow;
- a YAML/configuration mechanism if the Swin configuration style is used;
- optional nnU-Net, dynamic-network-architectures, MONAI, and related packages only if the project adopts the xLSTM-UNet framework.

The current environment has no importable PyTorch or torchvision. No compatibility fix or package installation is part of this pass.

## 9.4 Compatibility issues

- The official VisionLSTM2 code expects image-like input and performs its own patch embedding.
- The xLSTM-UNet custom ViL wrapper expects a sequence-compatible feature dimension and uses its own implementation.
- The xLSTM-UNet source is plan-driven and imports nnU-Net ecosystem components.
- The Swin-Unet source assumes fixed patch-grid resolutions and has source-specific decoder/pretraining behavior.
- The current Windows/Python 3.12 environment differs from the xLSTM-UNet README environment.

# 10. Open questions before implementation

## Data and spatial contract

- What exact Kvasir-SEG version will be used?
- What exact train/validation/test split will be frozen?
- Will images be directly resized or aspect-ratio padded?
- Is the final size `224x224`, `256x256`, or another value?
- What target dtype and binary loss will be used?

Evidence needed: a documented dataset/split decision and a data-contract review. No dataset download is required for this specification pass.

## Pure U-Net

- Should the baseline use plain convolution blocks or residual blocks?
- How many convolution layers per stage?
- InstanceNorm, BatchNorm, GroupNorm, or another normalization?
- LeakyReLU, ReLU, or another activation?
- Max pooling, stride-2 convolution, or another downsampling method?
- Nearest upsampling, transposed convolution, or another decoder method?
- Are the candidate widths appropriate for the available hardware?

Evidence needed: a project design decision; no reference implementation fixes these choices.

## ViL

- Is the first ViL experiment definitively the custom xLSTM-UNet block?
- Should one or more custom ViL blocks be stacked at the bottleneck?
- Should stochastic depth remain disabled initially?
- Should all ViL processing use float32, or should the later implementation preserve mixed precision where valid?
- Should any positional encoding be added as an ablation?
- Should official VisionLSTM2 be evaluated only as a separate model family?

Evidence needed: source-level adapter design and later forward-shape/compatibility tests.

## Multi-stage ViL

- Should the first multi-stage model use Stage 2 plus bottleneck, or Stage 3 plus bottleneck?
- Should each stage have separate ViL parameters?
- Should `C_s` equal the ViL dimension at every stage?
- Should stage outputs be saved before or after ViL for the experimental comparison?

Evidence needed: a frozen common-scaffold definition and resource measurements after implementation.

## Swin

- Should the comparator be native reference-faithful Swin-Unet or a common-scaffold Swin block model?
- How should native Swin channels `[96,192,384,768]` align with the chosen CNN decoder?
- Should the original patch-4 hierarchy be retained?
- Should any pretrained initialization be excluded or separately evaluated?

Evidence needed: a fairness decision and explicit adapter specification. No one-to-one equivalence should be assumed.

## Fairness and scale

- What tolerance defines approximately matched parameter count?
- Should matching prioritize parameters, FLOPs/MACs, activation memory, or all three?
- Which candidate scale is feasible on the available Colab runtime?
- How many random seeds are affordable for the final comparison?

Evidence needed: later model construction and measurement; no benchmark numbers are asserted here.

# Final classification of important decisions

| Decision | Classification | Basis |
|---|---|---|
| RGB input `[B,3,H,W]` | DIRECT SOURCE SUPPORT | Existing experiment protocol |
| Binary mask `[B,1,H,W]` | DIRECT SOURCE SUPPORT | Existing experiment protocol |
| Logit output `[B,1,H,W]` | DIRECT SOURCE SUPPORT | Existing experiment protocol |
| Fixed spatial size per run | STRONGLY SUPPORTED INFERENCE | Fixed-grid behavior of reference models |
| Starting resolution `224x224` | ENGINEERING CHOICE | Existing protocol recommendation, not frozen |
| Four-stage pure U-Net ladder | ENGINEERING CHOICE | No project CNN implementation fixes it |
| Pure U-Net widths `[32,64,128,256,256]` | ENGINEERING CHOICE | Lightweight provisional target |
| Pure U-Net normalization/activation | TBD | No project decision or standalone source baseline |
| NCHW to `[B,H*W,C]` flatten/restore | DIRECT SOURCE SUPPORT | xLSTM-UNet `ViLLayer` |
| Custom sequence-level ViL for first experiment | DIRECT SOURCE SUPPORT | Existing decision and xLSTM-UNet source |
| ViL dimension equals stage channels | STRONGLY SUPPORTED INFERENCE | xLSTM-UNet asserts channel/dim equality |
| Top-left traversal for first bottleneck | DIRECT SOURCE SUPPORT | xLSTM-UNet bottleneck wrapper |
| Avoid patch embedding inside first ViL block | STRONGLY SUPPORTED INFERENCE | Avoids second tokenization and matches custom wrapper |
| ViL after CNN stage/downsampling | STRONGLY SUPPORTED INFERENCE | xLSTM-UNet encoder execution order |
| Stage 2 plus bottleneck first multi-stage candidate | ENGINEERING CHOICE | Proposed resource-conscious progression |
| Native Swin-Unet resolution ladder | DIRECT SOURCE SUPPORT | Local Swin source/configuration |
| Common-scaffold Swin comparator | ENGINEERING CHOICE | Required for cleaner control, not original code |
| First model is Pure U-Net | DIRECT SOURCE SUPPORT | Decision log and frozen protocol |
| First controlled comparison from scratch | DIRECT SOURCE SUPPORT | Experiment protocol |
| Final hyperparameters and measurement budget | TBD | No project decision yet |

No novelty, superiority, or efficiency claim is made by this specification.
