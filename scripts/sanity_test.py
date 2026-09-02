"""Run a bounded end-to-end Pure U-Net integration sanity test."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sanity_pure_unet.json", type=Path)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="fail unless the selected runtime has a CUDA-capable PyTorch backend",
    )
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader
    except Exception as exc:
        print(f"BLOCKED: PyTorch is required for the runtime sanity test: {exc}")
        return 2

    from src.data.kvasir_seg import KvasirSegDataset
    from src.losses.segmentation import BCESoftDiceLoss
    from src.metrics.segmentation import batch_metric_values
    from src.models.pure_unet import PureUNet
    from src.training.checkpoint import load_checkpoint, save_checkpoint
    from src.training.config import choose_device, load_config, project_path
    from src.utils.reproducibility import (
        make_dataloader_generator,
        seed_everything,
        seed_worker,
    )

    config = load_config((PROJECT_ROOT / args.config).resolve())
    seed = int(config["seed"])
    seed_everything(seed)
    if args.require_cuda and not torch.cuda.is_available():
        print("FAIL: --require-cuda was requested, but CUDA is unavailable")
        return 3
    if args.require_cuda:
        config["device"] = "cuda"
    device = choose_device(config)
    dataset_config = config["dataset"]
    training_config = config["training"]
    dataset_kwargs = {
        "data_root": project_path(config, dataset_config["root"]),
        "manifest_path": project_path(config, dataset_config["manifest"]),
        "image_size": dataset_config["image_size"],
        "mask_threshold": dataset_config["mask_threshold"],
    }
    train_dataset = KvasirSegDataset(split="train", **dataset_kwargs)
    loader = DataLoader(
        train_dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        num_workers=int(training_config["num_workers"]),
        pin_memory=bool(training_config["pin_memory"]),
        worker_init_fn=seed_worker,
        generator=make_dataloader_generator(seed),
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = PureUNet(**config["model"]).to(device)
    criterion = BCESoftDiceLoss(**config["loss"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["scheduler"]["t_max"]),
        eta_min=float(config["scheduler"]["eta_min"]),
    )
    initial_lr = optimizer.param_groups[0]["lr"]
    first_batch = next(iter(loader))
    assert tuple(first_batch["image"].shape[1:]) == (3, 224, 224)
    assert tuple(first_batch["mask"].shape[1:]) == (1, 224, 224)
    assert first_batch["mask"].dtype == torch.float32
    assert set(first_batch["mask"].unique().tolist()).issubset({0.0, 1.0})

    metric_values = None
    output_shape = None
    test_start = time.perf_counter()
    batch_count = 0
    for batch_count, batch in enumerate(loader, start=1):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        assert tuple(logits.shape[1:]) == (1, 224, 224)
        loss = criterion(logits, masks)
        assert torch.isfinite(loss).item()
        loss.backward()
        optimizer.step()
        metric_values = batch_metric_values(logits.detach(), masks, threshold=0.5)
        output_shape = list(logits.shape)
        assert torch.isfinite(metric_values).all().item()
        if batch_count >= int(config["sanity"]["max_batches"]):
            break
    scheduler.step()
    stepped_lr = scheduler.get_last_lr()[0]
    checkpoint_path = project_path(config, config["sanity"]["checkpoint_path"])
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        epoch=1,
        validation_metrics={"dice": float(metric_values[:, 0].mean().item())},
        config=config,
        seed=seed,
    )
    restored = PureUNet(**config["model"]).to(device)
    load_checkpoint(checkpoint_path, restored, map_location=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - test_start
    gpu_name = None
    gpu_total_memory_bytes = None
    peak_gpu_memory_allocated_bytes = None
    peak_gpu_memory_reserved_bytes = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu_name = properties.name
        gpu_total_memory_bytes = int(properties.total_memory)
        peak_gpu_memory_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
        peak_gpu_memory_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
    print(
        json.dumps(
            {
                "status": "PASS",
                "device": str(device),
                "cuda_available": bool(torch.cuda.is_available()),
                "torch_cuda_version": torch.version.cuda,
                "gpu_name": gpu_name,
                "gpu_total_memory_bytes": gpu_total_memory_bytes,
                "peak_gpu_memory_allocated_bytes": peak_gpu_memory_allocated_bytes,
                "peak_gpu_memory_reserved_bytes": peak_gpu_memory_reserved_bytes,
                "batches": batch_count,
                "batch_size": int(training_config["batch_size"]),
                "input_shape": list(first_batch["image"].shape),
                "output_shape": output_shape,
                "elapsed_seconds": elapsed_seconds,
                "initial_lr": initial_lr,
                "lr_after_one_scheduler_step": stepped_lr,
                "checkpoint": str(checkpoint_path),
                "parameter_count": sum(p.numel() for p in model.parameters()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
