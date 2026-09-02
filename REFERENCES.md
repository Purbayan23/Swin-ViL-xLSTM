# Reference Registry

This registry records the reference material discovered during the project initialization and provenance pass on 2026-08-24. Paths are relative to the project root unless stated otherwise. Existing reference material is treated as immutable.

## Papers

### REF-PAPER-01

- Actual local path: `Reference/Papers/2105.05537v1.pdf`
- Title: *Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation*
- Authors: Hu Cao, Yueyue Wang, Joy Chen, Dongsheng Jiang, Xiaopeng Zhang, Qi Tian, Manning Wang
- Identifier: arXiv:2105.05537v1
- Version/date: v1, 2021-05-12, as printed in the PDF
- Upstream: https://arxiv.org/abs/2105.05537
- Role: hierarchical Swin Transformer U-Net segmentation reference
- Verification: VERIFIED from the PDF title page and local source README
- Notes: The local repository README identifies this paper and the ECCVW2022 medical computer vision workshop. No local Git metadata or license file was found.

### REF-PAPER-02

- Actual local path: `Reference/Papers/2405.04517v1.pdf`
- Title: *xLSTM: Extended Long Short-Term Memory*
- Authors: Maximilian Beck, Korbinian Pöppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michael Kopp, Günter Klambauer, Johannes Brandstetter, Sepp Hochreiter
- Identifier: arXiv:2405.04517v1
- Version/date: v1, 2024-05-07, as printed in the PDF
- Upstream: https://arxiv.org/abs/2405.04517
- Role: underlying xLSTM, sLSTM, and mLSTM mechanism reference
- Verification: VERIFIED from the PDF title page
- Notes: This paper is primarily about language modeling; its use here is as the underlying memory/block reference.

### REF-PAPER-03

- Actual local path: `Reference/Papers/2406.04303v2 Vision xLSTM.pdf`
- Title: *Vision-LSTM: xLSTM as Generic Vision Backbone*
- Authors: Benedikt Alkin, Maximilian Beck, Korbinian Pöppel, Sepp Hochreiter, Johannes Brandstetter
- Identifier: arXiv:2406.04303v2
- Version/date: v2, 2024-07-02, as printed in the PDF
- Upstream: https://arxiv.org/abs/2406.04303
- Role: Vision-LSTM/ViL visual token-processing reference
- Verification: VERIFIED from the PDF title page and local repository README
- Notes: The paper presents an isotropic visual backbone and does not provide the hierarchical U-Net encoder proposed as an optional project extension.

### REF-PAPER-04

- Actual local path: `Reference/Papers/2407.01530v2 ViL for segmention.pdf`
- Title: *xLSTM-UNet can be an Effective 2D & 3D Medical Image Segmentation Backbone with Vision-LSTM (ViL) better than its Mamba Counterpart*
- Authors: Tianrun Chen, Chaotao Ding, Lanyun Zhu, Tao Xu, Yan Wang, Deyi Ji, Ying Zang, Zejian Li
- Identifier: arXiv:2407.01530v2
- Version/date: v2, 2024-07-02, as printed in the PDF
- Upstream: https://arxiv.org/abs/2407.01530
- Role: medical segmentation integration reference for bottleneck and encoder ViL variants
- Verification: VERIFIED from the PDF title page and local repository README
- Notes: The source tree contains custom embedded ViL code rather than directly importing the current standalone VisionLSTM2 package.

## Repositories

### REF-REPO-01

- Actual local path: `Reference/repositories/Swin-Unet-main`
- Repository name: Swin-Unet
- Upstream URL: https://github.com/HuCaoFighting/Swin-Unet
- Relevant source: `networks/swin_transformer_unet_skip_expand_decoder_sys.py`, `networks/vision_transformer.py`
- Training/evaluation: `train.py`, `trainer.py`, `test.py`, shell scripts, YAML configuration
- Local source count: 19 files
- Commit: UNKNOWN; no `.git` directory is present
- Branch/tag: UNKNOWN; the directory suffix `-main` is not treated as verified branch metadata
- Dirty/clean status: UNKNOWN
- Approximate source date: UNKNOWN; README references the 2021 paper and ECCVW2022 workshop
- License: UNKNOWN; no license-like file was found
- Role: Swin-Unet architecture and training reference
- Verification: PROBABLE upstream identity; VERIFIED paper/code correspondence

### REF-REPO-02

- Actual local path: `Reference/repositories/vision-lstm-main`
- Repository name: Vision-LSTM (ViL)
- Upstream URL: https://github.com/NX-AI/vision-lstm
- Relevant source: `vision_lstm/vision_lstm2.py`, `vision_lstm/vision_lstm.py`, `vision_lstm/vision_lstm_util.py`
- Training/evaluation: `eval.py`, `src/`, `tutorials/`
- Local source count: 474 files
- Commit: UNKNOWN; no `.git` directory is present
- Branch/tag: UNKNOWN; the directory suffix `-main` is not treated as verified branch metadata
- Dirty/clean status: UNKNOWN
- Approximate source date: UNKNOWN; README explicitly supports VisionLSTM2 and links arXiv:2406.04303
- License: VERIFIED as mixed licensing from the README: MIT generally, with Apache-2.0 for `vision_lstm/vision_lstm.py`, `vision_lstm/vision_lstm2.py`, and `src/vislstm/modules/xlstm`
- Role: official Vision-LSTM implementation and CIFAR reference
- Verification: VERIFIED repository identity and notebook API correspondence

### REF-REPO-03

- Actual local path: `Reference/repositories/xLSTM-UNet-PyTorch-main`
- Repository name: xLSTM-UNet-PyTorch / U-xLSTM
- Upstream URL: https://github.com/tianrun-chen/xLSTM-UNet-PyTorch
- Relevant source: `UxLSTM/nnunetv2/nets/UxLSTMBot_2d.py`, `UxLSTM/nnunetv2/nets/UxLSTMBot_3d.py`, `UxLSTM/nnunetv2/nets/UxLSTMEnc_2d.py`, `UxLSTM/nnunetv2/nets/UxLSTMEnc_3d.py`, `UxLSTM/nnunetv2/nets/vision_lstm.py`
- Training/evaluation: custom nnU-Net trainers, `train_bot.sh`, `train_enc.sh`, `metric_bot.sh`, `metric_enc.sh`
- Local source count: 204 files
- Package version: `2.1.1` in `UxLSTM/setup.py`; this is the embedded nnU-Net package version, not a verified repository commit
- Commit: UNKNOWN; no `.git` directory is present
- Branch/tag: UNKNOWN; the directory suffix `-main` is not treated as verified branch metadata
- Dirty/clean status: UNKNOWN
- License: Apache License 2.0 file detected at `UxLSTM/LICENSE`; scope for the entire exported repository is UNKNOWN
- Role: medical segmentation integration reference for custom ViL bottleneck and encoder variants
- Verification: VERIFIED repository identity and paper correspondence

## Notebook

### REF-NB-01

- Actual local path: `Reference/notebooks/cifar10.ipynb`
- Name: CIFAR-10 Vision-LSTM reference notebook
- Format: Jupyter notebook, nbformat 4, 12 cells
- API: `from vision_lstm import VisionLSTM2`
- Configuration: `dim=192`, `depth=12`, `patch_size=4`, `input_shape=(3,32,32)`, `output_shape=(10,)`, `drop_path_rate=0.0`
- Role: static reference for official VisionLSTM2 usage
- Verification: VERIFIED by static inspection; not executed or modified
- Repository correspondence: VERIFIED against `Reference/repositories/vision-lstm-main/vision_lstm/vision_lstm2.py` and the repository README

## Other reference documents

### REF-BLOG-01

- Actual local path: `Reference/blogs/Swin_Amaarora.txt`
- Content: supplementary URL to https://amaarora.github.io/posts/2022-07-04-swintransformerv1.html
- Role: explanatory material only
- Verification: VERIFIED as a URL note; not used as the primary source for project decisions

### REF-PROTOCOL-01

- Actual local path: `Reference/Experiment_protocols/initial_frozen_protocol.txt`
- Name: initial frozen experimental protocol
- Role: existing project-direction record used to establish the planned progression and non-goals
- Verification: VERIFIED as an existing local protocol; not modified

## Provenance limitations

- The three repository trees are exported copies, not local Git clones.
- Local commits, branches, tags, remotes, and dirty/clean states are therefore UNKNOWN.
- No Kvasir-SEG dataset, split file, checkpoint, or training result is present in the reference tree.
- The exact date at which each repository export was created is UNKNOWN.
