from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ElectroCom-61 YOLO predictions on local photos.")
    parser.add_argument("--source", default="data/input", help="Image file or directory to test.")
    parser.add_argument("--model", default="models/electrocom61/best.pt", help="Path to trained YOLO weights.")
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum detection confidence.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", default="auto", help="Device for Ultralytics, for example auto, cpu, or 0.")
    parser.add_argument("--project", default="runs/predict", help="Output project directory.")
    parser.add_argument("--name", default="electrocom61-test", help="Output run name.")
    parser.add_argument("--save-txt", action="store_true", help="Also save YOLO txt predictions.")
    parser.add_argument("--save-conf", action="store_true", help="Include confidence values in YOLO txt predictions.")
    parser.add_argument("--save-crop", action="store_true", help="Save cropped detections.")
    parser.add_argument("--summary-csv", default="summary.csv", help="CSV filename written inside the output directory.")
    return parser.parse_args()


def result_path(result: Any) -> str:
    path = getattr(result, "path", "")
    return str(path)


def class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def write_summary_csv(results: list[Any], output_file: Path) -> int:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    detection_count = 0

    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])

        for result in results:
            names = getattr(result, "names", {})
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = [round(float(value), 2) for value in box.xyxy[0].tolist()]
                writer.writerow(
                    [
                        result_path(result),
                        class_id,
                        class_name(names, class_id),
                        round(confidence, 4),
                        x1,
                        y1,
                        x2,
                        y2,
                    ]
                )
                detection_count += 1

    return detection_count


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    model_path = Path(args.model)

    if not source.exists():
        print(f"ERROR: source does not exist: {source}", file=sys.stderr)
        print("Put your photos in data/input or pass --source with a file/folder path.", file=sys.stderr)
        return 1

    if source.is_dir() and not any(path.suffix.lower() in IMAGE_EXTENSIONS for path in source.rglob("*")):
        print(f"ERROR: no images found in source directory: {source}", file=sys.stderr)
        print("Supported extensions: " + ", ".join(sorted(IMAGE_EXTENSIONS)), file=sys.stderr)
        return 1

    if not model_path.exists():
        print(f"ERROR: model weights not found: {model_path}", file=sys.stderr)
        print("Expected trained weights at models/electrocom61/best.pt.", file=sys.stderr)
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics is not installed. Run: pip install -r requirements.txt", file=sys.stderr)
        return 1

    predict_args: dict[str, Any] = {
        "source": str(source),
        "conf": args.conf,
        "imgsz": args.imgsz,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
        "save": True,
        "save_txt": args.save_txt,
        "save_conf": args.save_conf,
        "save_crop": args.save_crop,
    }
    if args.device.lower() != "auto":
        predict_args["device"] = args.device

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    results = model.predict(**predict_args)

    output_dir = Path(args.project) / args.name
    detection_count = write_summary_csv(list(results), output_dir / args.summary_csv)

    print(f"Output directory: {output_dir}")
    print(f"Detections written to CSV: {detection_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
