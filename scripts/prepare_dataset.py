from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_PATHS = {
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
}
LEGACY_PATHS = {
    "../train/images": "train/images",
    "../valid/images": "valid/images",
    "../test/images": "test/images",
}


@dataclass
class SplitSummary:
    name: str
    image_count: int
    label_count: int
    box_count: int
    empty_label_count: int


class DatasetError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fix and validate an ElectroCom-61 YOLO dataset.")
    parser.add_argument("--dataset", default="data_t/ElectroCom-61_v2", help="Dataset directory containing data.yaml.")
    parser.add_argument("--fix", action="store_true", help="Rewrite data.yaml when Roboflow paths need normalization.")
    parser.add_argument("--expected-imgsz", type=int, default=640, help="Expected square image size. Mismatches are warnings.")
    return parser.parse_args()


def normalize_yolo_path(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    normalized = value.replace("\\", "/").strip()
    return LEGACY_PATHS.get(normalized, normalized)


def load_and_fix_yaml(data_yaml: Path, fix: bool) -> dict[str, Any]:
    if not data_yaml.exists():
        raise DatasetError(f"Missing data.yaml: {data_yaml}")

    with data_yaml.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise DatasetError("data.yaml must contain a YAML mapping.")

    changed = False
    for key, expected in EXPECTED_PATHS.items():
        if key not in data:
            raise DatasetError(f"data.yaml is missing required key: {key}")
        normalized = normalize_yolo_path(data[key])
        if normalized != data[key]:
            data[key] = normalized
            changed = True
        if normalized != expected:
            print(f"WARNING: data.yaml key '{key}' is '{normalized}', expected '{expected}'.")

    names = data.get("names")
    if not isinstance(names, list) or not names:
        raise DatasetError("data.yaml must contain a non-empty 'names' list.")

    nc = data.get("nc")
    if nc is None:
        data["nc"] = len(names)
        changed = True
    elif int(nc) != len(names):
        raise DatasetError(f"data.yaml nc={nc} but names contains {len(names)} classes.")

    if changed:
        if not fix:
            raise DatasetError("data.yaml needs path fixes. Re-run with --fix.")
        with data_yaml.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
        print(f"Updated {data_yaml}")
    else:
        print("data.yaml is already normalized.")

    return data


def resolve_split_dir(dataset_dir: Path, relative_path: str) -> Path:
    return (dataset_dir / relative_path).resolve()


def validate_image(image_path: Path, expected_imgsz: int) -> tuple[bool, str | None]:
    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            width, height = image.size
    except Exception as exc:  # Pillow raises several concrete exception types.
        return False, f"{image_path}: unreadable image ({exc})"

    if width != expected_imgsz or height != expected_imgsz:
        return True, f"{image_path}: image size is {width}x{height}, expected {expected_imgsz}x{expected_imgsz}"
    return True, None


def validate_label_file(label_path: Path, class_count: int) -> tuple[int, bool, list[str], Counter[int]]:
    errors: list[str] = []
    class_counter: Counter[int] = Counter()
    box_count = 0

    raw_lines = label_path.read_text(encoding="utf-8").splitlines()
    non_empty_lines = [line.strip() for line in raw_lines if line.strip()]

    for line_number, line in enumerate(non_empty_lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{label_path}:{line_number}: expected 5 YOLO fields, got {len(parts)}")
            continue
        try:
            class_id = int(float(parts[0]))
            values = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"{label_path}:{line_number}: non-numeric YOLO field")
            continue
        if class_id < 0 or class_id >= class_count:
            errors.append(f"{label_path}:{line_number}: class id {class_id} outside 0..{class_count - 1}")
        if any(value < 0.0 or value > 1.0 for value in values):
            errors.append(f"{label_path}:{line_number}: bbox values must be normalized between 0 and 1")
        box_count += 1
        class_counter[class_id] += 1

    return box_count, len(non_empty_lines) == 0, errors, class_counter


def validate_split(dataset_dir: Path, split_key: str, image_relative_path: str, class_count: int, expected_imgsz: int) -> tuple[SplitSummary, Counter[int], list[str], list[str]]:
    image_dir = resolve_split_dir(dataset_dir, image_relative_path)
    split_dir = image_dir.parent
    label_dir = split_dir / "labels"

    errors: list[str] = []
    warnings: list[str] = []
    class_counter: Counter[int] = Counter()
    total_boxes = 0
    empty_labels = 0

    if not image_dir.exists():
        raise DatasetError(f"Missing {split_key} image directory: {image_dir}")
    if not label_dir.exists():
        raise DatasetError(f"Missing {split_key} label directory: {label_dir}")

    images = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    labels = sorted(path for path in label_dir.glob("*.txt") if path.is_file())
    label_by_stem = {path.stem: path for path in labels}
    image_stems = {path.stem for path in images}

    for image_path in images:
        ok, warning = validate_image(image_path, expected_imgsz)
        if not ok:
            errors.append(warning or f"{image_path}: unreadable image")
        elif warning:
            warnings.append(warning)

        label_path = label_by_stem.get(image_path.stem)
        if label_path is None:
            errors.append(f"{image_path}: missing matching label file")
            continue

        box_count, is_empty, label_errors, split_counts = validate_label_file(label_path, class_count)
        total_boxes += box_count
        empty_labels += int(is_empty)
        errors.extend(label_errors)
        class_counter.update(split_counts)

    extra_labels = [path for path in labels if path.stem not in image_stems]
    if extra_labels:
        warnings.append(f"{split_key}: {len(extra_labels)} label file(s) do not have a matching image.")

    summary = SplitSummary(
        name=split_key,
        image_count=len(images),
        label_count=len(labels),
        box_count=total_boxes,
        empty_label_count=empty_labels,
    )
    return summary, class_counter, errors, warnings


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset).resolve()
    data_yaml = dataset_dir / "data.yaml"

    try:
        if not dataset_dir.exists():
            raise DatasetError(f"Dataset directory does not exist: {dataset_dir}")

        data = load_and_fix_yaml(data_yaml, fix=args.fix)
        class_names = data["names"]
        class_count = len(class_names)

        all_counts: Counter[int] = Counter()
        all_errors: list[str] = []
        all_warnings: list[str] = []
        summaries: list[SplitSummary] = []

        for split_key in ("train", "val", "test"):
            summary, counts, errors, warnings = validate_split(
                dataset_dir=dataset_dir,
                split_key=split_key,
                image_relative_path=normalize_yolo_path(data[split_key]),
                class_count=class_count,
                expected_imgsz=args.expected_imgsz,
            )
            summaries.append(summary)
            all_counts.update(counts)
            all_errors.extend(errors)
            all_warnings.extend(warnings)

        print("\nDataset summary")
        for summary in summaries:
            print(
                f"- {summary.name}: {summary.image_count} images, "
                f"{summary.label_count} labels, {summary.box_count} boxes, "
                f"{summary.empty_label_count} empty labels"
            )
        print(f"- classes: {class_count}")
        print(f"- total boxes: {sum(all_counts.values())}")

        missing_classes = [index for index in range(class_count) if all_counts[index] == 0]
        if missing_classes:
            all_warnings.append(f"{len(missing_classes)} class(es) have zero boxes: {missing_classes}")

        if all_warnings:
            print("\nWarnings")
            for warning in all_warnings[:25]:
                print(f"- {warning}")
            if len(all_warnings) > 25:
                print(f"- ... {len(all_warnings) - 25} more warning(s)")

        if all_errors:
            print("\nErrors", file=sys.stderr)
            for error in all_errors[:50]:
                print(f"- {error}", file=sys.stderr)
            if len(all_errors) > 50:
                print(f"- ... {len(all_errors) - 50} more error(s)", file=sys.stderr)
            return 1

        print("\nDataset validation passed.")
        return 0
    except DatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
