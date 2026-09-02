# Project_ViL

Project_ViL is a research-oriented, computationally lightweight project for studying Vision-LSTM-based feature processing against CNN and Swin Transformer feature processing within a 2D U-Net segmentation framework.

## Current status

The Pure U-Net baseline is implemented and has passed the bounded CPU sanity test. The local laptop/Codex environment is used for CPU development and debugging; Google Colab is reserved for GPU sanity testing and the later experiments. Full 100-epoch training has not been run.

## Reference structure

- `Reference/Papers/` - source PDFs
- `Reference/repositories/` - immutable exported source trees
- `Reference/notebooks/` - immutable reference notebook
- `Reference/Experiment_protocols/` - existing project protocol material
- `Reference/blogs/` - supplementary explanatory material
- `Reference/audits/` - provenance and audit documents created by this project
- `architecture-reference/` - reserved for architecture reference material; diagrams are intentionally postponed

## Planned progression

1. Pure CNN U-Net baseline
2. U-Net with a ViL bottleneck
3. U-Net with ViL at selected deeper encoder stages
4. Lightweight Swin comparison
5. Optional hierarchical ViL extension

The minimal Colab workflow is documented in `COLAB_WORKFLOW.md`. It copies the persistent Drive dataset once to `/content/kvasir-seg`, reads the local copy during training, and writes checkpoints/results back to Drive.

The project is experimental and educational. It does not claim novelty, state-of-the-art performance, or inherent ViL superiority or efficiency.
