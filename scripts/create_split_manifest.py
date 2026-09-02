"""Verify Kvasir-SEG pairs and create the frozen seed-42 split manifest."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageChops


def verify_pairs(data_root: Path) -> list[dict[str, str]]:
    image_dir = data_root / "images"
    mask_dir = data_root / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"expected images/ and masks/ under {data_root}")

    image_files = sorted(path for path in image_dir.iterdir() if path.is_file())
    mask_files = sorted(path for path in mask_dir.iterdir() if path.is_file())
    image_by_stem = {path.stem: path for path in image_files}
    mask_by_stem = {path.stem: path for path in mask_files}
    missing_masks = sorted(set(image_by_stem) - set(mask_by_stem))
    missing_images = sorted(set(mask_by_stem) - set(image_by_stem))
    if missing_masks or missing_images:
        raise RuntimeError(
            f"unpaired files: missing_masks={missing_masks[:5]}, "
            f"missing_images={missing_images[:5]}"
        )

    pairs: list[dict[str, str]] = []
    for sample_id in sorted(image_by_stem):
        image_path = image_by_stem[sample_id]
        mask_path = mask_by_stem[sample_id]
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != mask.size:
                raise RuntimeError(
                    f"dimension mismatch for {sample_id}: {image.size} vs {mask.size}"
                )
            if image.mode != "RGB" or mask.mode != "RGB":
                raise RuntimeError(
                    f"unexpected modes for {sample_id}: {image.mode} vs {mask.mode}"
                )
            red, green, blue = mask.split()
            if ImageChops.difference(red, green).getbbox() is not None or ImageChops.difference(
                red, blue
            ).getbbox() is not None:
                raise RuntimeError(f"mask RGB channels differ for {sample_id}")
        pairs.append(
            {
                "id": sample_id,
                "image": image_path.relative_to(data_root).as_posix(),
                "mask": mask_path.relative_to(data_root).as_posix(),
            }
        )
    return pairs


def make_manifest(data_root: Path, output_path: Path, seed: int) -> dict:
    pairs = verify_pairs(data_root)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    count = len(pairs)
    train_count = int(count * 0.70)
    validation_count = int(count * 0.15)
    manifest = {
        "schema_version": 1,
        "dataset": "Kvasir-SEG",
        "dataset_root": "data/Kvasir-SEG",
        "pairing": "matched image/mask filename stems",
        "split_policy": {
            "seed": seed,
            "ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "ordering": "lexicographically sorted stems, then random.Random(seed).shuffle",
        },
        "preprocessing_note": {
            "mask_threshold": 128,
            "threshold_status": "engineering choice, not an official Kvasir-SEG threshold",
            "mask_resize": "nearest",
        },
        "counts": {
            "total": count,
            "train": train_count,
            "validation": validation_count,
            "test": count - train_count - validation_count,
        },
        "splits": {
            "train": pairs[:train_count],
            "validation": pairs[train_count : train_count + validation_count],
            "test": pairs[train_count + validation_count :],
        },
    }
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/Kvasir-SEG", type=Path)
    parser.add_argument(
        "--output",
        default="data/splits/kvasir_seg_seed42_70_15_15.json",
        type=Path,
    )
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(
            f"refusing to overwrite frozen manifest: {output_path}; "
            "remove it only after an explicit protocol decision"
        )
    manifest = make_manifest(data_root, output_path, args.seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps(manifest["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
