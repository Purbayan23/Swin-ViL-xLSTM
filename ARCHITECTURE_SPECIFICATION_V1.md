# Architecture Specification V1

Status: architecture specification and implementation record. The Pure U-Net baseline and corrected Architecture A0 are complete and frozen. Architecture A1 is the current independent traversal ablation; its implementation and bounded CPU sanity validation are complete, but the first full run failed at epoch 27 and is not a valid completed result. A read-only numerical-stability investigation is recorded below; no architecture or protocol change has been approved.

Date: 2026-09-03

This document converts the current project protocol and the inspected reference source trees into a tensor-level implementation target. It deliberately separates source-backed facts from engineering recommendations and unresolved decisions.

The Pure U-Net choices for the first experiment are frozen in `BASELINE_SPECIFICATION_V1.md`. Corrected Architecture A0 changes only the bottleneck feature-processing mechanism; A1 changes only the directional traversal mechanism relative to A0. Their encoder, decoder, skips, data contract, loss, optimizer, scheduler, and run protocol remain shared with the baseline. A1 is an independent ablation, not a stacked adaptation.

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

The first baseline and Architecture A use the frozen `224x224` resolution. Alternate resolutions or aspect-ratio-preserving preprocessing remain separate future experiments.

Classification:

- Fixed-size processing within a run: **STRONGLY SUPPORTED INFERENCE**.
- Starting at `224x224`: **ENGINEERING CHOICE** frozen in the baseline protocol.
- Alternate image resolution and aspect-ratio policy: **TBD** for future experiments.

## 1.3 Target and output tensors

```text
Y:       [B, 1, H, W]   binary target mask
logits:  [B, 1, H, W]   raw foreground logits
```

The target mask represents background/foreground using values `{0, 1}`. The output is one foreground logit channel; a separate background logit is not required by this contract.

Classification:

- Shape and binary-mask contract: **DIRECT SOURCE SUPPORT** from `EXPERIMENT_PROTOCOL.md`.
- Target tensor dtype: **ENGINEERING CHOICE**; the implemented baseline uses float32 binary masks for BCE plus soft Dice.
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
- Exact fusion operation and decoder channels: **ENGINEERING CHOICE** frozen by the implemented baseline and reused by Architecture A.

# 2. Pure U-Net baseline

## 2.1 Reference status

No standalone project CNN U-Net implementation is present in `Reference/`. The xLSTM-UNet repository contains `UNetResEncoder` and `UNetResDecoder`, but these are nnU-Net-derived components with plan-driven widths, normalization, strides, and deep supervision. Reusing them would import additional framework behavior into the first baseline.

Therefore, the pure U-Net architecture below is a project design target, not a directly copied reference implementation.

Classification: **DIRECT SOURCE SUPPORT** that the pure U-Net must be the first model; its implemented details are **ENGINEERING CHOICES** recorded in `BASELINE_SPECIFICATION_V1.md`.

## 2.2 Implemented V1 tensor ladder

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

This is the implemented and frozen first-baseline ladder. Architecture A reuses it unchanged.

Classification:

- Four encoder stages plus one bottleneck: **ENGINEERING CHOICE**.
- Candidate widths `[32, 64, 128, 256, 256]`: **ENGINEERING CHOICE**.
- Resolution ladder at `224x224`: **ENGINEERING CHOICE** based on the existing protocol recommendation.
- Final number of stages, widths, and input resolution for V1: **ENGINEERING CHOICE** implemented and frozen; later scale variants remain open.

## 2.3 Encoder blocks

Implemented encoder stage:

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
- Two `3x3` convolutions per stage: **ENGINEERING CHOICE** frozen in the baseline.
- Plain convolution block, InstanceNorm, and LeakyReLU: **ENGINEERING CHOICE** frozen in the baseline.
- MaxPool downsampling: **ENGINEERING CHOICE** frozen in the baseline.

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
- Number and type of Pure U-Net CNN bottleneck blocks: **ENGINEERING CHOICE** frozen in the baseline; Architecture A adds the separate ViL/mLSTM processor after this CNN block.

## 2.5 Decoder

Implemented decoder contract:

```text
upsample to the next skip resolution
→ explicitly align channels
→ concatenate or otherwise fuse the matching skip
→ convolution block(s)
```

The decoder must reverse the encoder spatial ladder and produce `[B, C_0, H, W]` before the segmentation head.

Classification:

- U-Net-style mirrored decoder and skip fusion: **DIRECT SOURCE SUPPORT** at the architectural level.
- Bilinear interpolation plus `1x1` projection: **ENGINEERING CHOICE** frozen in the baseline.
- The decoder uses the same two-convolution InstanceNorm/LeakyReLU blocks and concatenated skips as the Pure U-Net baseline.

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
- The implemented head uses a one-channel `1x1` convolution with raw logits; sigmoid is applied only for loss/metrics: **ENGINEERING CHOICE** frozen in the baseline.

# 3. ViL bottleneck

## 3.1 Selected ViL implementation

Architecture A uses the custom sequence-level implementation pattern in:

```text
Reference/repositories/xLSTM-UNet-PyTorch-main/
UxLSTM/nnunetv2/nets/UxLSTMBot_2d.py
```

That wrapper calls the custom `ViLBlock` in:

```text
Reference/repositories/xLSTM-UNet-PyTorch-main/
UxLSTM/nnunetv2/nets/vision_lstm.py
```

It is not the complete official `VisionLSTM2` class. The project-local implementation is `src/models/vil_bottleneck_unet.py` and does not import the reference repositories.

Classification: **DIRECT SOURCE SUPPORT** for the integration pattern; the project-local PyTorch port is an **ENGINEERING CHOICE**.

## 3.2 Exact tensor transformation

At the implemented Pure U-Net bottleneck, `F_b` is `[B,256,14,14]`, so `H_b*W_b=196` and the token embedding dimension is `C_b=256`:

```text
F_b                    [B, 256, 14, 14]
flatten spatial order  [B, 196, 256]
ViL/mLSTM block        [B, 196, 256]
restore spatial order  [B, 256, 14, 14]
```

The flattening is row-major: for each batch item, height rows are traversed from top to bottom and columns within each row from left to right. The inverse transpose/reshape restores the same coordinate mapping.

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
- `D=C_b` at the external block interface and no external projections: **STRONGLY SUPPORTED INFERENCE** and implemented engineering choice.

## 3.3 Normalization and residual structure

The inspected custom `ViLBlock` contains:

```text
LayerNorm
→ custom ViLLayer
→ DropPath residual wrapper
```

The custom `ViLLayer` itself contains the mLSTM projections, causal 1D convolution, matrix-LSTM cell, learnable skip inside the mLSTM branch, gating, and output projection.

The xLSTM-UNet wrapper also defines a `self.norm`, but its forward method calls `self.vil` directly; the effective normalization for this path is the LayerNorm inside `ViLBlock`.

The project adapter uses one pre-normalized residual ViL/mLSTM block. Its residual is the direct `x + block(LayerNorm(x))` path corresponding to the reference `DropPath` with `drop_path=0`. The mLSTM branch also retains the reference learnable skip before output gating. No additional external residual or normalization layer is added.

Classification: **DIRECT SOURCE SUPPORT** for the inspected custom block; deterministic residual/no-drop-path behavior is an **ENGINEERING CHOICE** matching the reference default.

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

For the first bottleneck, the implementation:

- avoids patch embedding inside ViL;
- avoids additional VisionLSTM2 positional embedding;
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

## 3.7 Architecture A0 implementation record

The corrected A0 implementation uses one custom ViL/mLSTM block with the following configuration:

| Component | Configuration |
|---|---|
| External input/output | `[B,256,14,14]` |
| Sequence length | `196` |
| External/token embedding dimension | `256` |
| Block depth | `1` |
| Internal projection dimension | `512` from expansion `2` |
| Q/K/V projection grouping | `128` heads of dimension `4` |
| Matrix-LSTM grouping | `4` heads of dimension `128` |
| Local context convolution | causal depthwise 1-D convolution, kernel size `4` |
| Traversal | one top-left-to-bottom-right row-major sequence |
| Normalization | residual-weight LayerNorm before the block; per-head output normalization |
| Projection biases | disabled for ViL projections, as in the adapted reference defaults |
| Convolution bias | enabled for the causal depthwise convolution |
| Stochastic depth | disabled |
| Positional/patch embedding | none |

The parallel matrix-memory computation follows the stabilized mLSTM equations in the inspected xLSTM-UNet `vision_lstm.py`. The reference deliberately uses different groupings for the headwise Q/K/V projections and the matrix-LSTM cell: with inner dimension `512` and `qkv_block_size=4`, Q/K/V use `128` groups of width `4`, while the matrix-LSTM uses `4` heads of width `128`. Native PyTorch reshaping, `einsum`, grouped convolution, and normalization replace the reference `einops` and framework-specific imports.

The A0 constructor initializes the shared Pure U-Net modules first and attaches the additional ViL/mLSTM processor afterward. This preserves the exact Pure U-Net parameter initialization under a shared seed; the processor consumes only subsequent random draws. Corresponding Pure U-Net parameters match exactly under seed `42`.

A0 passed the bounded CPU sanity path and focused unit tests. Its corrected full 100-epoch Colab experiment completed with the best validation-Dice checkpoint at epoch `51`.

### A0 completed result

| Quantity | Recorded value |
|---|---:|
| Best validation Dice | `0.8005566217` at epoch 51 |
| Test Dice | `0.8175639115` |
| Test IoU | `0.7270158563` |
| Test precision | `0.8651886918` |
| Test recall | `0.8180166952` |

A0 is frozen. The earlier incorrect-head, initialization-confounded epoch-78 run remains historical and incomplete; it is not a valid A0 result.

## 3.8 Architecture A1: alternating bidirectional bottleneck ablation

A1 is introduced before Architecture B to test whether the spatial traversal mechanism affects the bottleneck result. It is an independent controlled ablation of frozen A0, not an additional layer stacked on top of A0.

The external tensor contract is unchanged:

```text
[B,256,14,14]
→ [B,196,256]
→ alternating ViL/mLSTM pair
→ [B,196,256]
→ [B,256,14,14]
```

For each A1 pair, let `X` be the row-major token sequence. The first block processes the original order:

```text
Y_forward = Block_top_left_to_bottom_right(X)
```

The second, independently parameterized block processes the reverse order and is then aligned back to the original spatial positions:

```text
Y_reverse = flip(
    Block_bottom_right_to_top_left(flip(Y_forward, sequence_dimension)),
    sequence_dimension
)
```

The pair output is `Y_reverse`. This is sequential composition, matching the cited Vision-LSTM `ViLBlockPair`; it is not averaging, concatenation, or parameter sharing. Each block retains its own pre-normalized residual, causal mLSTM, internal learnable skip, gating, and output projection. Both directions therefore have embedding dimension `256`, internal projection dimension `512`, Q/K/V grouping `128 x 4`, and matrix-LSTM grouping `4 x 128`. Positional encoding and patch embedding remain excluded.

The reverse flip is applied before the second block and undone immediately afterward, so every output token remains aligned with its original `(row, column)` location before restoration to `[B,256,14,14]`. With one independent directional pair, the derived A1 parameter count is `4,814,945 + 2 x 415,496 = 5,645,937`.

The A1 pair mechanism is source-derived from the Vision-LSTM `ViLBlockPair` and its explicit top-left/bottom-right traversal. The use of that pair at the existing xLSTM-UNet-pattern bottleneck, while retaining the project’s causal 1-D convolution and all A0 controls, is a project-specific engineering adaptation.

## 3.9 A1 numerical-stability incident record

The first full A1 run followed the approved A1 configuration through epoch 26 and then produced `NaN` training and validation losses at epoch 27. Dice and IoU were reported as zero afterward. The best validation Dice observed before failure was `0.693657` at epoch 15. This run is failed/incomplete and must not be used as an experimental result; its historical log/checkpoint metadata is retained.

The failed Colab epoch-26 state was not present in the checked-out project, so the exact first non-finite operation in that remote run could not be conclusively localized. Runtime-only inspection of the current implementation found a latent shared mLSTM hazard in `_parallel_stabilized_mlstm`: `torch.exp(-max_log_decay)` can overflow in float32 when `max_log_decay < -88.722839`. A controlled finite-input stress case produced a finite mLSTM output and finite loss, followed first by non-finite input-gate gradients, then non-finite AdamW state and parameters. The same hazard exists in A0’s single block, but A1 adds a second independently parameterized mLSTM path and may encounter the condition there first; this remains unproven for the failed run.

On the local seed-42 real-data trace, A1 and A0 had finite activations, losses, gradients, parameters, and optimizer states. The causal-mask `-inf` entries were expected masked values. The bounded fixed-batch A1 trace did not reproduce the failure. No clipping, epsilon, precision, learning-rate, architecture, or protocol change was made. The next required evidence is an instrumented replay from the available epoch-26 Colab state, if retained.

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

The following remain provisional project scale envelopes, not official named model variants. The completed Pure U-Net and implemented Architecture A parameter counts are recorded below; future scale variants and multi-stage placements remain unimplemented.

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

## 6.3 Implemented parameter counts

| Model | Trainable parameters | Difference from Pure U-Net |
|---|---:|---:|
| Pure U-Net | `4,814,945` | — |
| Architecture A0: Pure U-Net + one ViL/mLSTM bottleneck block | `5,230,441` | `+415,496` |
| Architecture A1: Pure U-Net + one independent alternating ViL/mLSTM pair | `5,645,937` | `+830,992` |

The increases are attributable to the project-local bottleneck processors. No parameter matching was imposed. A0 and A1 sanity configurations use batch size 1 only as bounded diagnostics; their full experiment configurations preserve the approved batch size 4. The earlier `5,611,617` count described the superseded 128-head matrix-LSTM implementation and is not a result for A0 or A1.

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

This recommendation was implemented and validated as the completed Pure U-Net baseline. It is:

- technically valid;
- the easiest model to debug;
- small relative to the source paper-scale models;
- independent of ViL and Swin-specific compatibility issues;
- required for validating the data, loss, metrics, checkpointing, and evaluation pipeline.

The following details were subsequently frozen in `BASELINE_SPECIFICATION_V1.md`:

- exact convolution block definition;
- normalization;
- activation;
- downsampling operator;
- decoder upsampling operator;
- exact width cap;
- exact input resize/aspect-ratio policy;
- loss and optimizer.

Corrected A0 is frozen after its completed 100-epoch experiment. A1 is implemented and bounded-sanity validated, but its first full experiment failed at epoch 27 and is not a valid result. Architecture B remains deferred until the A1 numerical-stability issue is localized and the traversal ablation can be evaluated validly.

Classification: **DIRECT SOURCE SUPPORT** for the first-model ordering; the concrete Pure U-Net and Architecture A project choices are **ENGINEERING CHOICES** recorded in the baseline and implementation configurations.

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

The current local environment has importable PyTorch `2.6.0+cpu`, NumPy, and Pillow. Architecture A uses only native PyTorch operations and therefore does not require `einops`, torchvision, nnU-Net, dynamic-network-architectures, or another compiled extension. No package installation was required for this implementation pass.

## 9.4 Compatibility issues

- The official VisionLSTM2 code expects image-like input and performs its own patch embedding.
- The xLSTM-UNet custom ViL wrapper expects a sequence-compatible feature dimension and uses its own implementation.
- The xLSTM-UNet source is plan-driven and imports nnU-Net ecosystem components.
- The Swin-Unet source assumes fixed patch-grid resolutions and has source-specific decoder/pretraining behavior.
- The current Windows/Python 3.12 environment differs from the xLSTM-UNet README environment.

# 10. Open questions before implementation

## Data and spatial contract

- The first baseline uses the frozen seed-42 `70/15/15` split and direct `224x224` preprocessing documented in `BASELINE_SPECIFICATION_V1.md`.
- The exact Kvasir-SEG release/version identifier remains TBD for provenance.
- Aspect-ratio-preserving preprocessing, alternate resolutions, and target/loss variants are future controlled experiments.

Evidence needed: a documented dataset/split decision and a data-contract review. No dataset download is required for this specification pass.

## Pure U-Net

- The first baseline uses the plain two-convolution Conv–InstanceNorm–LeakyReLU blocks, MaxPool downsampling, bilinear-plus-`1x1` decoder, and `32-64-128-256-256` widths documented in `BASELINE_SPECIFICATION_V1.md`.
- Alternate blocks, widths, decoder operators, and resolutions are separate future experiments.

Evidence needed: a project design decision; no reference implementation fixes these choices.

## ViL

- The first ViL implementation is resolved as the custom xLSTM-UNet block adapted into project-local native PyTorch code; future alternative blocks remain open.
- Corrected A0 uses one custom ViL/mLSTM block at the bottleneck. A1 uses one independent alternating directional pair at the same bottleneck; it is not A0 stacked with another adaptation.
- Stochastic depth is disabled for Architecture A; later regularization remains open.
- The current adapter promotes float16/bfloat16 inputs to float32 for numerical stability; a later mixed-precision design remains open.
- Positional encoding is omitted from Architecture A; adding it is a separate ablation.
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
| Starting resolution `224x224` | ENGINEERING CHOICE | Frozen first-baseline configuration |
| Four-stage pure U-Net ladder | ENGINEERING CHOICE | No project CNN implementation fixes it |
| Pure U-Net widths `[32,64,128,256,256]` | ENGINEERING CHOICE | Lightweight provisional target |
| Pure U-Net normalization/activation | ENGINEERING CHOICE | Frozen in `BASELINE_SPECIFICATION_V1.md` |
| NCHW to `[B,H*W,C]` flatten/restore | DIRECT SOURCE SUPPORT | xLSTM-UNet `ViLLayer` |
| Custom sequence-level ViL for first experiment | DIRECT SOURCE SUPPORT | Existing decision and xLSTM-UNet source |
| ViL dimension equals stage channels | STRONGLY SUPPORTED INFERENCE | xLSTM-UNet asserts channel/dim equality |
| Top-left traversal for first bottleneck | DIRECT SOURCE SUPPORT | xLSTM-UNet bottleneck wrapper |
| One top-left-to-bottom-right block for Architecture A | ENGINEERING CHOICE | Matches the xLSTM-UNet bottleneck wrapper |
| Internal expansion `2`, qkv block size `4`, kernel size `4` | ENGINEERING CHOICE | Adapted reference defaults |
| ViL residual and mLSTM learnable skip | ENGINEERING CHOICE | Adapted from xLSTM-UNet `ViLBlock` and `ViLLayer` |
| Avoid patch embedding inside first ViL block | STRONGLY SUPPORTED INFERENCE | Avoids second tokenization and matches custom wrapper |
| No additional positional encoding | STRONGLY SUPPORTED INFERENCE | Custom xLSTM-UNet bottleneck receives CNN features |
| ViL after CNN stage/downsampling | STRONGLY SUPPORTED INFERENCE | xLSTM-UNet encoder execution order |
| Stage 2 plus bottleneck first multi-stage candidate | ENGINEERING CHOICE | Proposed resource-conscious progression |
| Native Swin-Unet resolution ladder | DIRECT SOURCE SUPPORT | Local Swin source/configuration |
| Common-scaffold Swin comparator | ENGINEERING CHOICE | Required for cleaner control, not original code |
| First model is Pure U-Net | DIRECT SOURCE SUPPORT | Decision log and frozen protocol |
| First controlled comparison from scratch | DIRECT SOURCE SUPPORT | Experiment protocol |
| First-baseline hyperparameters and measurement budget | ENGINEERING CHOICE | Frozen in `BASELINE_SPECIFICATION_V1.md`; future changes require a new decision |

No novelty, superiority, or efficiency claim is made by this specification.
