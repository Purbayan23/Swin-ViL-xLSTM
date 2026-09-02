# Project_ViL

Project_ViL is a research-oriented, computationally lightweight project for studying Vision-LSTM-based feature processing against CNN and Swin Transformer feature processing within a 2D U-Net segmentation framework.

## Current status

The first Pure U-Net baseline experiment is complete. The implementation passed bounded CPU and Colab GPU sanity tests, including the CUDA test on an NVIDIA Tesla T4. The completed Colab run used the frozen Kvasir-SEG split and the approved 100-epoch protocol; the best validation-Dice checkpoint was selected at epoch 40.

The best validation Dice was `0.8284274113` at epoch 40. Reported test metrics for the completed run are Dice `0.8241105953`, IoU `0.7412125480`, precision `0.8742652225`, and recall `0.8361236785`. These values are recorded for the baseline run only and are not claims of novelty or state-of-the-art performance. Training loss continued decreasing after approximately epoch 40 while validation performance degraded, supporting best-checkpoint selection.

The baseline uses 1,000 verified Kvasir-SEG image/mask pairs with a deterministic seed-42 70/15/15 split (700/150/150). Images and masks are resized to `224×224`; masks are converted to grayscale, thresholded with the engineering choice `gray >= 128` before nearest-neighbor resizing, and returned as binary tensors. The model is a Pure U-Net with widths `32-64-128-256-256`, BCE + soft Dice loss, AdamW with learning rate `1e-3`, and cosine annealing over 100 epochs.

The local laptop/Codex environment is used for CPU development and debugging, while Google Colab is used for GPU experiments. The reproducible qualitative post-hoc workflow is provided by `scripts/visualize_predictions.py`; it creates a deterministic eight-image fixed-sample grid and a separate grid of the four lowest-Dice test cases without changing training or checkpoint selection. The difficult cases showed localization errors, false-positive over-segmentation, near-complete misses, and incomplete capture of larger targets. These are baseline observations, not claims that a future ViL model will solve them.

## Reference structure

- `Reference/Papers/` - source PDFs
- `Reference/repositories/` - immutable exported source trees
- `Reference/notebooks/` - immutable reference notebook
- `Reference/Experiment_protocols/` - existing project protocol material
- `Reference/blogs/` - supplementary explanatory material
- `Reference/audits/` - provenance and audit documents created by this project
- `architecture-reference/` - reserved for architecture reference material; diagrams are intentionally postponed

## Planned progression

1. Pure CNN U-Net baseline (completed and frozen)
2. Architecture A: Pure U-Net + ViL/mLSTM bottleneck (next controlled experiment)
3. U-Net with ViL at selected deeper encoder stages
4. Lightweight Swin comparison
5. Optional hierarchical ViL extension

The minimal Colab workflow is documented in `COLAB_WORKFLOW.md`. It copies the persistent Drive dataset once to `/content/kvasir-seg`, reads the local copy during training, and writes checkpoints/results back to Drive. The baseline configuration is `configs/colab_pure_unet.json`; qualitative predictions can be generated with:

```bash
python scripts/visualize_predictions.py --config configs/colab_pure_unet.json
```

The project is experimental and educational. It does not claim novelty, state-of-the-art performance, or inherent ViL superiority or efficiency.
