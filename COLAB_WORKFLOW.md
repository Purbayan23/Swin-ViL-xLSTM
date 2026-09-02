# Google Colab Workflow

The laptop/Codex environment is for CPU development and debugging. Google Colab is the GPU environment for the bounded GPU sanity test and later experiments. Colab's working PyTorch installation is used; the project does not pin a CUDA wheel for Colab.

Google Drive is persistent storage. `/content` is temporary session-local storage. The dataset is copied once from Drive to `/content/kvasir-seg` before verification and training; training reads images and masks from that local copy. Checkpoints and results are written to Drive.

## Fresh runtime

The repository is `https://github.com/Purbayan23/Swin-ViL-xLSTM`.

```python
!git clone https://github.com/Purbayan23/Swin-ViL-xLSTM /content/Project_ViL
%cd /content/Project_ViL

from google.colab import drive
drive.mount('/content/drive')

!python scripts/prepare_colab_dataset.py
!python scripts/environment_check.py --require-cuda
!python scripts/sanity_test.py --config configs/colab_sanity_pure_unet.json --require-cuda
```

Run the full baseline only after the sanity command passes:

```python
!python scripts/train.py --config configs/colab_pure_unet.json
!python scripts/evaluate.py --config configs/colab_pure_unet.json
```

The sanity command is bounded to two batches. It does not run the 100-epoch experiment.

The bounded CUDA sanity test has passed on an NVIDIA Tesla T4. Rerun it in each fresh Colab runtime to verify the active environment before training.

If `/content/kvasir-seg` already exists, the preparation command refuses to silently reuse or replace it. Use `--reuse` only after accepting the verified local copy, or use `--replace` to explicitly rebuild the session-local copy.

## Drive layout

```text
/content/drive/MyDrive/Project_ViL/
├── data/
│   └── Kvasir-SEG/
│       ├── images/
│       └── masks/
└── experiments/
    ├── sanity/
    └── runs/
```

The frozen split manifest remains in the cloned repository at `data/splits/kvasir_seg_seed42_70_15_15.json`. The Colab configurations point image/mask loading to `/content/kvasir-seg` and checkpoints/results to Drive without changing Python source code.
