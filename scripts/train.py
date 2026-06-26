from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on ElectroCom-61 and export best.pt.")
    parser.add_argument("--data", default="data_t/ElectroCom-61_v2/data.yaml", help="Path to data.yaml.")
    parser.add_argument("--model", default="yolo11s.pt", help="Ultralytics model checkpoint.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", default="auto")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="electrocom61-v1")
    parser.add_argument("--copy-best-to", default="models/electrocom61/best.pt")
    parser.add_argument("--resume", action="store_true", help="Resume from project/name/weights/last.pt.")
    return parser.parse_args()


def parse_batch(value: str) -> str | int | float:
    if value.lower() == "auto":
        return "auto"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid batch value: {value}") from exc


def copy_best_weight(run_dir: Path, destination: Path) -> None:
    best_weight = run_dir / "weights" / "best.pt"
    if not best_weight.exists():
        raise FileNotFoundError(f"Training finished but best.pt was not found: {best_weight}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weight, destination)
    print(f"Copied best model to {destination}")


def main() -> int:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics is not installed. Run start_training.ps1 first.", file=sys.stderr)
        return 1

    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        print(f"ERROR: data.yaml does not exist: {data_yaml}", file=sys.stderr)
        return 1

    project_dir = Path(args.project)
    run_dir = project_dir / args.name
    destination = Path(args.copy_best_to)

    if args.resume:
        last_weight = run_dir / "weights" / "last.pt"
        if not last_weight.exists():
            print(f"ERROR: cannot resume because last.pt does not exist: {last_weight}", file=sys.stderr)
            return 1

        print(f"Resuming training from {last_weight}")
        model = YOLO(str(last_weight))
        model.train(resume=True, device=args.device)
    else:
        train_args: dict[str, Any] = {
            "data": str(data_yaml),
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "device": args.device,
            "batch": parse_batch(args.batch),
            "project": str(project_dir),
            "name": args.name,
            "exist_ok": True,
        }

        print(
            "Training YOLO with "
            f"model={args.model}, data={data_yaml}, epochs={args.epochs}, "
            f"imgsz={args.imgsz}, device={args.device}, batch={args.batch}"
        )
        model = YOLO(args.model)
        model.train(**train_args)

    copy_best_weight(run_dir, destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
