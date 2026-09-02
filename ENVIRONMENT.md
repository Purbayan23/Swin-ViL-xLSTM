# Environment Record

This record describes the current environment observed during the implementation verification pass on 2026-09-02. The approved CPU-only PyTorch installation was completed after the initial missing-PyTorch audit.

## Detected values

- Operating system: Microsoft Windows NT 10.0.26200.0
- Python executable inspected: `C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`
- Python version: 3.12.13
- pip version: 26.2.1
- Git version: 2.55.0.windows.4
- GPU detected by `nvidia-smi`: NVIDIA GeForce MX330
- GPU driver: 581.95
- GPU memory reported by `nvidia-smi`: 2048 MiB
- NumPy: importable in the inspected runtime, version 2.3.5
- pypdf: importable in the inspected runtime
- pdfplumber: importable in the inspected runtime
- Pillow/PIL: importable in the inspected runtime, version 12.3.0
- PyTorch: importable in the inspected runtime, version 2.6.0+cpu
- PyTorch CUDA availability: `False` (CPU-only build intentionally selected)
- PyTorch CUDA runtime: none (CPU-only build)

## Unavailable in the inspected runtime

- torchvision: unavailable and not required by the implementation
- torchaudio: unavailable and not required by the implementation
- NVIDIA driver reports CUDA Version 13.0; CUDA toolkit installation was not independently verified
- PyYAML: unavailable
- einops: unavailable
- MONAI: unavailable
- nnU-Net/nnunetv2: unavailable

## Implementation verification (2026-09-02)

- Python executable: `C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`
- Python: 3.12.13
- PyTorch: 2.6.0+cpu; import and CPU tensor operation passed
- `torch.cuda.is_available()`: `False`, as expected for the CPU-only build
- torchvision: missing and not required by the implementation
- PyYAML: missing and not required; configuration uses standard-library JSON
- tqdm: missing and not required
- CUDA hardware visibility: `nvidia-smi` detects an NVIDIA GeForce MX330 with 2,048 MiB and driver-reported CUDA 13.0
- Approved installation: `torch==2.6.0` from the official CPU wheel index; no CUDA PyTorch was installed
- PyTorch transitive runtime dependencies installed by pip: `filelock`, `networkx`, `jinja2`, `fsspec`, `sympy`, `mpmath`, and `MarkupSafe`; `typing-extensions` and `setuptools` were already present
- Bounded sanity test: passed for 2 batches on CPU; parameter count `4,814,945`; checkpoint save/reload passed
- Full 100-epoch training: not run

## Repository status

- Project root: no local `.git` directory was detected.
- Reference repository exports: no local `.git` directory was detected in any of the three exported source trees.
- Reference repository remotes, branches, commits, tags, and dirty/clean status: UNKNOWN.

## Compatibility note

The xLSTM-UNet README describes an Ubuntu 20.04, Python 3.10, CUDA 11.8, PyTorch 2.0.1 environment. The current local environment is Windows with Python 3.12.13 and PyTorch 2.6.0+cpu. This is recorded for provenance only; no reference-repository compatibility changes were attempted.

## Local and Colab environment separation

- Local laptop/Codex: Windows, Python 3.12.13, PyTorch 2.6.0+cpu; used for implementation and CPU debugging.
- Google Colab: intended GPU environment for the bounded GPU sanity test and later experiments. Colab's working PyTorch/CUDA environment should be inspected by `scripts/environment_check.py`; it is not assumed to match the local runtime.
- The project uses relative repository paths for the local configuration and a separate Drive-path configuration for Colab. No local CUDA installation is required.

## Colab storage workflow

- Google Drive is persistent storage for the Kvasir-SEG source and experiment outputs.
- `/content` is session-local storage. `scripts/prepare_colab_dataset.py` verifies the Drive source, copies it once to `/content/kvasir-seg`, and verifies the local copy.
- Colab training reads the dataset from `/content/kvasir-seg`, not from mounted Drive. Colab checkpoints, training history, evaluation results, and metadata are written under `/content/drive/MyDrive/Project_ViL/experiments/`.
- The Colab workflow has not been executed in this local environment.
