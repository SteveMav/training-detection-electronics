from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check PyTorch CUDA availability.")
    parser.add_argument("--require-cuda", action="store_true", help="Exit with an error when CUDA is not available.")
    parser.add_argument("--device", default="0", help="CUDA device index expected by Ultralytics, usually 0.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is not installed. Run start_training.ps1 to install dependencies.", file=sys.stderr)
        return 1

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")

    if not torch.cuda.is_available():
        if args.require_cuda:
            print("ERROR: CUDA is required but PyTorch cannot access it.", file=sys.stderr)
            print("Check NVIDIA drivers with nvidia-smi and install a CUDA-enabled PyTorch build.", file=sys.stderr)
            return 1
        return 0

    device_index = 0
    if str(args.device).lower() not in {"cpu", "mps"}:
        try:
            device_index = int(str(args.device).split(",")[0])
        except ValueError:
            print(f"ERROR: Unsupported device value for CUDA check: {args.device}", file=sys.stderr)
            return 1

    device_count = torch.cuda.device_count()
    print(f"CUDA device count: {device_count}")

    if device_index < 0 or device_index >= device_count:
        print(f"ERROR: Requested CUDA device {device_index}, but only {device_count} device(s) are visible.", file=sys.stderr)
        return 1

    props = torch.cuda.get_device_properties(device_index)
    total_gb = props.total_memory / (1024**3)
    print(f"Selected GPU: cuda:{device_index} - {props.name} ({total_gb:.1f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
