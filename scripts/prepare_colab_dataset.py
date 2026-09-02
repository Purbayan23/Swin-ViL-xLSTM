"""Copy and verify Kvasir-SEG from Google Drive to Colab local storage."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


EXPECTED_PAIR_COUNT = 1000
IMAGE_SUFFIXES = {".jpg", ".jpeg"}


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def verify_dataset(root: Path) -> dict[str, object]:
    root = root.resolve()
    image_dir = root / "images"
    mask_dir = root / "masks"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"images directory is missing: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"masks directory is missing: {mask_dir}")

    image_files = _image_files(image_dir)
    mask_files = _image_files(mask_dir)
    if len(image_files) != EXPECTED_PAIR_COUNT:
        raise ValueError(f"expected {EXPECTED_PAIR_COUNT} images, found {len(image_files)}")
    if len(mask_files) != EXPECTED_PAIR_COUNT:
        raise ValueError(f"expected {EXPECTED_PAIR_COUNT} masks, found {len(mask_files)}")

    image_stems = [path.stem for path in image_files]
    mask_stems = [path.stem for path in mask_files]
    if len(set(image_stems)) != len(image_stems):
        raise ValueError("duplicate image filename stems detected")
    if len(set(mask_stems)) != len(mask_stems):
        raise ValueError("duplicate mask filename stems detected")
    image_ids = set(image_stems)
    mask_ids = set(mask_stems)
    missing_masks = sorted(image_ids - mask_ids)
    missing_images = sorted(mask_ids - image_ids)
    if missing_masks or missing_images:
        raise ValueError(
            f"image/mask stem mismatch: missing_masks={missing_masks[:5]}, "
            f"missing_images={missing_images[:5]}"
        )

    boxes_path = root / "kavsir_bboxes.json"
    return {
        "root": str(root),
        "image_count": len(image_files),
        "mask_count": len(mask_files),
        "paired_count": len(image_ids),
        "missing_masks": missing_masks,
        "missing_images": missing_images,
        "bounding_boxes_present": boxes_path.is_file(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/content/drive/MyDrive/Project_ViL/data/Kvasir-SEG"),
    )
    parser.add_argument("--destination-root", type=Path, default=Path("/content/kvasir-seg"))
    existing = parser.add_mutually_exclusive_group()
    existing.add_argument(
        "--reuse",
        action="store_true",
        help="verify and reuse an existing destination instead of copying",
    )
    existing.add_argument(
        "--replace",
        action="store_true",
        help="remove the existing destination before copying",
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    destination_root = args.destination_root.resolve()
    if destination_root == source_root:
        raise ValueError("source and destination roots must be different")
    try:
        destination_root.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise ValueError("destination must not be inside the source dataset")
    source_summary = verify_dataset(source_root)

    if destination_root.exists():
        if args.reuse:
            destination_summary = verify_dataset(destination_root)
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "action": "reused",
                        "source": source_summary,
                        "destination": destination_summary,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if not args.replace:
            raise FileExistsError(
                f"destination already exists: {destination_root}; "
                "use --reuse after verification or --replace explicitly"
            )
        if destination_root == Path(destination_root.anchor):
            raise ValueError("refusing to replace a filesystem root")
        shutil.rmtree(destination_root)

    destination_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_root, destination_root)
    destination_summary = verify_dataset(destination_root)
    print(
        json.dumps(
            {
                "status": "PASS",
                "action": "copied",
                "source": source_summary,
                "destination": destination_summary,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
