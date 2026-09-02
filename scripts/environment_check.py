"""Report the runtime dependencies and accelerator visible to the project."""

from __future__ import annotations

import argparse
import json
import platform
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="return a non-zero status unless CUDA is available through PyTorch",
    )
    args = parser.parse_args()

    report = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }
    try:
        import numpy as np
        from PIL import __version__ as pillow_version
        import torch
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = str(exc)
        print(json.dumps(report, sort_keys=True))
        return 2

    cuda_available = bool(torch.cuda.is_available())
    report.update(
        {
            "status": "PASS" if not args.require_cuda or cuda_available else "FAIL",
            "torch": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": cuda_available,
            "numpy": np.__version__,
            "pillow": pillow_version,
        }
    )
    if cuda_available:
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        report.update(
            {
                "gpu_name": properties.name,
                "gpu_compute_capability": list(torch.cuda.get_device_capability(device)),
                "gpu_total_memory_bytes": int(properties.total_memory),
            }
        )
    else:
        report.update(
            {
                "gpu_name": None,
                "gpu_compute_capability": None,
                "gpu_total_memory_bytes": None,
            }
        )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
