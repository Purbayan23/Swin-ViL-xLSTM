# Decision Log

Dates use the project date available when each entry was recorded. Initialization entries are dated 2026-08-24; later implementation and result entries retain their recorded dates. These entries preserve historical decisions and do not replace earlier records.

## D-001 - Use Kvasir-SEG as the primary dataset

- Date: 2026-08-24
- Decision: Use Kvasir-SEG rather than ISIC for the primary experiment.
- Rationale: It is a feasible 2D binary polyp-segmentation dataset for the lightweight project scope and provides pixel-level masks.
- Evidence: `Reference/Experiment_protocols/initial_frozen_protocol.txt`; Kvasir-SEG source information reviewed during the prior audit.
- Status: Established; the first-baseline split was subsequently frozen in D-012; the exact dataset release identifier remains TBD.

## D-002 - Do not pursue novelty or SOTA

- Date: 2026-08-24
- Decision: Treat the project as an experimental/educational PhD-interview project rather than a novelty or SOTA claim.
- Rationale: The purpose is to compare feature-processing mechanisms under controlled conditions.
- Evidence: Project prompt and existing frozen protocol.
- Status: Established.

## D-003 - Keep raw source repositories immutable

- Date: 2026-08-24
- Decision: Keep the three exported source repositories under `Reference/repositories/` and do not modify them.
- Rationale: Reference code is evidence and provenance, not project implementation code.
- Evidence: Existing frozen protocol and initialization instructions.
- Status: Established.

## D-004 - Keep the supplied notebook immutable

- Date: 2026-08-24
- Decision: Preserve `Reference/notebooks/cifar10.ipynb` as a static reference and do not execute or modify it.
- Rationale: It documents official VisionLSTM2 usage while remaining separate from future project notebooks.
- Evidence: Static notebook inspection and existing frozen protocol.
- Status: Established.

## D-005 - Do not treat VisionLSTM2 as a direct Swin-block replacement

- Date: 2026-08-24
- Decision: VisionLSTM2 is not considered a one-to-one Swin Transformer block replacement.
- Rationale: It expects image-like input, performs its own patch embedding and positional encoding, and does not natively expose a hierarchical U-Net skip structure.
- Evidence: Local Vision-LSTM source tree and prior architecture audit.
- Status: Established.

## D-006 - Start with a pure U-Net baseline

- Date: 2026-08-24
- Decision: Validate a pure CNN U-Net baseline before ViL or Swin experiments.
- Rationale: The dataset, preprocessing, loss, metrics, checkpointing, and evaluation pipeline must be validated independently of the research modules.
- Evidence: Existing frozen protocol.
- Status: Established planned progression.

## D-007 - First ViL experiment is bottleneck integration

- Date: 2026-08-24
- Decision: First ViL experiment will process the deepest U-Net feature as a flattened spatial sequence and reshape it back before decoding.
- Rationale: It is the simplest controlled insertion and closely corresponds to xLSTM-UNet_Bot.
- Evidence: xLSTM-UNet paper and local source tree; prior architecture audit.
- Status: Established direction; exact block configuration TBD.

## D-008 - Test deeper/multi-stage ViL later

- Date: 2026-08-24
- Decision: Test ViL at selected deeper encoder stages only after the bottleneck experiment is validated.
- Rationale: This follows the xLSTM-UNet_Enc direction while preserving a staged experimental progression.
- Evidence: xLSTM-UNet paper and local source tree.
- Status: Established planned progression.

## D-009 - Add Swin comparison after basic validation

- Date: 2026-08-24
- Decision: Add the lightweight Swin comparison after the CNN and initial ViL pipeline are validated.
- Rationale: A common scaffold and data protocol should be working before introducing the additional Swin-specific variables.
- Evidence: Existing frozen protocol and prior audit.
- Status: Established planned progression.

## D-010 - Defer hierarchical ViL

- Date: 2026-08-24
- Decision: Treat hierarchical ViL as optional and do not implement it prematurely.
- Rationale: Vision-LSTM is presented as isotropic, while a hierarchical ViL encoder requires additional wrappers and design decisions.
- Evidence: Vision-LSTM paper and prior architecture audit.
- Status: Established deferral.

## D-011 - Postpone architecture diagrams

- Date: 2026-08-24
- Decision: Do not create architecture diagrams during initialization.
- Rationale: The architecture is not fully frozen and the initialization pass is documentation/provenance only.
- Evidence: Initialization instructions.
- Status: Established.

## D-012 - Freeze the first Kvasir-SEG split

- Date: 2026-09-02
- Decision: Use the locally verified 1,000 paired samples with a deterministic 70%/15%/15% filename-stem split, seed 42, frozen in `data/splits/kvasir_seg_seed42_70_15_15.json`.
- Rationale: The local dataset has complete image/mask pairing and no verified subject/video grouping metadata used by this project.
- Evidence: Read-only local dataset verification and manifest generation.
- Status: Active for the first baseline; future models must reuse it.

## D-013 - Canonicalize compressed Kvasir-SEG masks before resizing

- Date: 2026-09-02
- Decision: Convert RGB masks to grayscale, apply `gray >= 128`, convert to binary, then resize with nearest-neighbor to 224x224.
- Rationale: The local masks are RGB JPEGs with identical channels and separated decoded values around black (`0–8`) and white (`246–255`).
- Evidence: Local dataset verification; Kvasir-SEG source documentation describes white polyp foreground and black background; project protocol requires nearest-neighbor mask resizing.
- Status: Active implementation rule. The value 128 is an engineering choice, not an official Kvasir-SEG threshold.

## D-014 - Implement the Pure U-Net baseline before future models

- Date: 2026-09-02
- Decision: Implement only the configuration-driven Pure U-Net baseline and bounded sanity-test entry point; do not run the full 100-epoch experiment in this pass.
- Rationale: Validate the common data, loss, metric, optimizer, scheduler, and checkpoint pipeline before adding ViL or Swin.
- Evidence: `BASELINE_SPECIFICATION_V1.md` and the first-baseline implementation files.
- Status: Implemented; the initial runtime check was blocked before the approved PyTorch installation. Superseded by D-015 for current runtime status.

## D-015 - Complete local and Colab runtime verification

- Date: 2026-09-02
- Decision: Record the approved local CPU runtime and the subsequent Colab GPU sanity verification as complete. Local PyTorch 2.6.0+cpu was installed and the bounded Pure U-Net sanity test passed. The Colab CUDA sanity test subsequently passed on an NVIDIA Tesla T4 using Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8, and 15.6 GB GPU memory.
- Rationale: The data, model, optimization, checkpoint, and evaluation pipeline has now passed bounded verification in the intended local development and Colab GPU environments before the first full experiment.
- Evidence: Local bounded CPU sanity-test output and the recorded Colab Tesla T4 CUDA sanity-test result.
- Status: Complete for runtime verification; the full 100-epoch Pure U-Net experiment remains not run.

## D-016 - Freeze the completed Pure U-Net baseline and proceed to Architecture A

- Date: 2026-09-03
- Decision: Freeze the completed Pure U-Net baseline as the first experimental reference and proceed to the independent comparison **Architecture A: Pure U-Net + ViL/mLSTM bottleneck**.
- Rationale: The full 100-epoch Kvasir-SEG run completed under the approved protocol, with the best validation-Dice checkpoint selected at epoch 40. The baseline establishes an operational end-to-end pipeline, meaningful validation-based checkpoint selection, successful learning on many cases, heterogeneous performance, and interpretable localization, false-positive, and under-segmentation failure modes.
- Evidence: Completed Colab baseline run; best validation Dice `0.828427411334084`; test Dice `0.8241105952570058`; test IoU `0.7412125480073589`; post-hoc fixed-sample and four-lowest-Dice qualitative analysis.
- Controlled-comparison rule: Reuse the dataset, frozen seed-42 split, preprocessing, evaluation metrics, and standardized training protocol. Architecture A will use the planned sequence-level pattern `[B,C,H,W] -> [B,H*W,C] -> ViL/mLSTM feature block -> [B,H*W,C] -> [B,C,H,W]` and is not implemented by this decision entry.
- Status: Baseline frozen; next experiment authorized as a separately implemented controlled comparison. The four lowest-Dice cases are post-hoc analysis only and are not representative random samples or tuning data.

## D-017 - Implement Architecture A as the xLSTM-UNet-pattern bottleneck adapter

- Date: 2026-09-03
- Decision: Implement Architecture A as a project-local, sequence-level ViL/mLSTM bottleneck while changing no component of the frozen Pure U-Net pipeline outside the bottleneck feature processor.
- Rationale: The xLSTM-UNet 2D bottleneck wrapper directly provides the required feature-map-to-sequence-to-feature-map integration. The complete official VisionLSTM2 model is not a drop-in block because it includes image patch embedding, positional embedding, and an alternating block-pair backbone.
- Initial implementation record (head semantics superseded by D-018): `[B,256,14,14] -> [B,196,256] -> one top-left-to-bottom-right ViL/mLSTM block -> [B,196,256] -> [B,256,14,14]`, with internal expansion 2, the then-used 128-head matrix-LSTM grouping, causal depthwise kernel size 4, residual path, and no patch or positional embedding.
- Evidence: `ARCHITECTURE_SPECIFICATION_V1.md`; `Reference/repositories/xLSTM-UNet-PyTorch-main/UxLSTM/nnunetv2/nets/UxLSTMBot_2d.py`; `Reference/repositories/xLSTM-UNet-PyTorch-main/UxLSTM/nnunetv2/nets/vision_lstm.py`; cited Vision-LSTM and xLSTM-UNet papers; bounded CPU sanity and focused unit tests.
- Historical parameter effect before the D-018 correction: Pure U-Net `4,814,945`; Architecture A `5,611,617`; increase `796,672`. This superseded implementation was not treated as a completed experiment.
- Dependencies: Native PyTorch implementation; no `einops`, torchvision, nnU-Net, dynamic-network-architectures, compiled extension, or package installation required.
- Status: Initial implementation record retained for chronology; superseded in part by D-018. Full Architecture A training has not been run.

## D-018 - Correct Architecture A head semantics and initialization control

- Date: 2026-09-03
- Decision: Correct the project-local Architecture A adapter to preserve the cited xLSTM-UNet grouping: Q/K/V projections use `128` heads of dimension `4`, while the matrix-LSTM cell uses `4` heads of dimension `128` for the expanded dimension `512`.
- Rationale: The reference `vision_lstm.py` computes Q/K/V projection heads as `inner_dim // qkv_block_size` but constructs `MatrixLSTMCell(..., num_heads=qkv_block_size)`. The previous implementation reused the Q/K/V head count for the matrix-LSTM cell and therefore did not reproduce the cited semantics.
- Initialization policy: Construct the shared Pure U-Net superclass first, then attach the additional ViL/mLSTM processor. This preserves the exact Pure U-Net CNN parameter initialization under seed `42`; the processor consumes subsequent random draws independently.
- Parameter effect: Corrected Architecture A has `5,230,441` parameters, consisting of the Pure U-Net `4,814,945` plus `415,496` processor parameters.
- Scope: Dataset, preprocessing, split, traversal, causal 1-D convolution, decoder, loss, optimizer, scheduler, batch size, epoch budget, seed, checkpoint rule, and prediction threshold are unchanged. The interrupted epoch-78 training history/checkpoint metadata is historical and is not reclassified as a completed experiment.
- Evidence: Reference `xLSTM-UNet-PyTorch-main/UxLSTM/nnunetv2/nets/vision_lstm.py`; corrected project implementation; focused CPU tests and bounded CPU sanity test.
- Status: Corrected implementation; full Architecture A training remains not run.
