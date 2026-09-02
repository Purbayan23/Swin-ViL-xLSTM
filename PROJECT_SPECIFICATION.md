# Project Specification

## Status

This document records the project contract and its implementation status. The first Pure U-Net baseline is implemented; the full 100-epoch experiment has not been run.

## Purpose

Project_ViL is an experimental and educational medical-image segmentation project intended to support a future PhD interview. It investigates feature-processing mechanisms within a lightweight 2D U-Net framework.

## Research question

> How does Vision-LSTM-based feature processing compare with CNN and Swin Transformer feature processing within a lightweight U-Net framework for 2D medical-image segmentation?

## Non-goals

The project does not claim:

- a novel architecture;
- state-of-the-art performance;
- that ViL is inherently better than Swin;
- that ViL is inherently faster or more efficient;
- general superiority across medical segmentation tasks.

## Primary dataset

- Dataset: Kvasir-SEG
- Task: binary 2D polyp segmentation
- Input: RGB endoscopy image
- Target: binary pixel-level mask
- Dataset acquisition: local `data/Kvasir-SEG/` copy verified
- Dataset split: frozen `data/splits/kvasir_seg_seed42_70_15_15.json` with 700/150/150 pairs

## Planned model progression

1. Pure CNN U-Net baseline.
2. U-Net with a ViL bottleneck.
3. U-Net with ViL at selected deeper encoder stages.
4. Lightweight Swin-based comparison.
5. Optional hierarchical ViL extension after the earlier experiments are validated.

## Computational constraint

The project targets a basic laptop and Google Colab. The initial scope is lightweight 2D experimentation. 3D segmentation, full paper-scale pretraining, and large benchmark reproduction are explicitly deferred.

## Scientific philosophy

The project will use a common data and training protocol where practical, document unavoidable architecture-specific differences, measure computation rather than assume it, and restrict conclusions to the evaluated dataset and experiments.

## Baseline implementation status

- Configuration: `configs/baseline_pure_unet.json`
- Sanity configuration: `configs/sanity_pure_unet.json`
- Mask preprocessing: grayscale, `gray >= 128`, binary conversion, then nearest-neighbor resize; 128 is an engineering choice, not an official dataset threshold.
- Full training: not run; PyTorch is missing from the inspected runtime.

## Open specification items

- Exact image resolution: `224x224` for the first baseline; future resolution changes are separate experiments.
- Exact CNN channel schedule: `32-64-128-256-256` for the first baseline.
- Exact ViL block source/configuration: TBD within the documented custom-block versus official-VisionLSTM2 decision.
- Exact Swin configuration: TBD.
- Exact split, optimizer, scheduler, loss, batch size, update budget, and checkpoint rule: resolved for the first baseline in `BASELINE_SPECIFICATION_V1.md`; future comparison-specific changes remain out of scope.
