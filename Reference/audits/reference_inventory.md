# Reference Inventory Audit

Audit date: 2026-08-24

## Scope

This inventory covers the actual local `Reference/` directory. The project uses the capitalized directory name `Reference`; no lower-case `references/` directory was created or used merely to match earlier naming.

Existing papers, repositories, notebook, protocol, and blog note were read or statically inspected without modification. The notebook was not executed.

## Top-level reference tree

```text
Reference/
├── audits/
├── blogs/
├── Experiment_protocols/
├── notebooks/
├── Papers/
└── repositories/
```

## Papers found

| Actual path | Type | Intended reference | Status |
|---|---|---|---|
| `Reference/Papers/2105.05537v1.pdf` | PDF | Swin-Unet | VERIFIED |
| `Reference/Papers/2405.04517v1.pdf` | PDF | xLSTM | VERIFIED |
| `Reference/Papers/2406.04303v2 Vision xLSTM.pdf` | PDF | Vision-LSTM | VERIFIED |
| `Reference/Papers/2407.01530v2 ViL for segmention.pdf` | PDF | xLSTM-UNet | VERIFIED |

The four PDFs contain the expected arXiv identifiers and title-page bibliographic information. Their exact local filenames were preserved.

## Repository trees found

| Actual path | Type | Likely source | Status |
|---|---|---|---|
| `Reference/repositories/Swin-Unet-main` | Exported Python source tree | HuCaoFighting/Swin-Unet | PROBABLE identity; paper correspondence VERIFIED |
| `Reference/repositories/vision-lstm-main` | Exported Python source tree and training code | NX-AI/vision-lstm | VERIFIED |
| `Reference/repositories/xLSTM-UNet-PyTorch-main` | Exported Python source tree and nnU-Net package | tianrun-chen/xLSTM-UNet-PyTorch | VERIFIED |

All three directories contain source structures matching the intended projects. None contains a `.git` directory. Local commit, branch, tag, remote, and dirty/clean status are therefore UNKNOWN.

## Notebook found

| Actual path | Type | Intended reference | Status |
|---|---|---|---|
| `Reference/notebooks/cifar10.ipynb` | Jupyter notebook, nbformat 4 | Vision-LSTM CIFAR-10 example | VERIFIED |

Static inspection found `from vision_lstm import VisionLSTM2` with `dim=192`, `depth=12`, `patch_size=4`, `input_shape=(3,32,32)`, `output_shape=(10,)`, and `drop_path_rate=0.0`. The notebook also contains CIFAR-10 download/training cells, but it was not executed.

## Other reference documents

| Actual path | Type | Role | Status |
|---|---|---|---|
| `Reference/Experiment_protocols/initial_frozen_protocol.txt` | Plain text protocol | Existing project-direction record | VERIFIED; not modified |
| `Reference/blogs/Swin_Amaarora.txt` | URL note | Supplementary Swin explanation | VERIFIED; not used as primary evidence |
| `Reference/audits/` | Directory | Audit output location | Existing directory was empty before this inventory document |

## Repository consistency checks

### Swin-Unet

- The local README identifies the Swin-Unet paper arXiv:2105.05537.
- The local source contains `SwinTransformerSys`, `SwinTransformerBlock`, `WindowAttention`, `PatchMerging`, `PatchExpand`, and the encoder/decoder training entry points.
- The intended upstream GitHub URL is supplied by the project instructions and matches the repository identity.
- Upstream commit and export date: UNKNOWN.

### Vision-LSTM

- The local README links arXiv:2406.04303 and the NX-AI GitHub repository.
- The local source contains `VisionLSTM2`, `ViLBlockPair`, `MatrixLSTMCell`, patch embedding, and positional-embedding utilities.
- The notebook import and API match the local `vision_lstm` package.
- Mixed MIT/Apache licensing is explicitly described in the local README.
- Upstream commit and export date: UNKNOWN.

### xLSTM-UNet

- The local README links arXiv:2407.01530 and the GitHub clone URL.
- The local source contains bottleneck and encoder variants, 2D and 3D implementations, custom ViL code, nnU-Net trainers, and shell entry points.
- `UxLSTM/setup.py` reports package version `2.1.1`.
- The repository contains an Apache license file under `UxLSTM/`.
- Upstream commit and export date: UNKNOWN.

## Environment consistency

The current environment is Windows with Python 3.12.13, an NVIDIA GeForce MX330, and no importable PyTorch or torchvision. The xLSTM-UNet README describes a different Ubuntu/CUDA/PyTorch environment. No compatibility changes were attempted.

## Missing or unresolved information

- No local Git provenance for any exported repository.
- No verified repository commit or branch for any export.
- No Kvasir-SEG data, split, preprocessing artifact, or checkpoint.
- No final training hyperparameters.
- No final architecture configuration.
- Whole-repository license scope for Swin-Unet and xLSTM-UNet is not fully established from local license files.
- Exact repository export dates are UNKNOWN.

## Safety confirmation

No original paper, raw source tree, notebook, protocol, or blog file was modified, renamed, reformatted, or overwritten. This audit document is new project documentation.
