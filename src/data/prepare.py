from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASS_TO_LABEL = {"NORMAL": 0, "PNEUMONIA": 1}
SOURCE_SPLITS = ("train", "val", "test")
PATIENT_PATTERNS = (
    re.compile(r"^(person\d+)", re.IGNORECASE),
    re.compile(r"^(NORMAL2-IM-\d+)", re.IGNORECASE),
    re.compile(r"^(IM-\d+)", re.IGNORECASE),
)


def locate_dataset_root(input_root: Path) -> Path:
    """Locate the directory containing train/val/test folders."""

    input_root = input_root.expanduser().resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {input_root}")

    candidates = [
        input_root,
        input_root / "chest_xray",
        input_root / "chest_xray" / "chest_xray",
    ]

    for train_dir in input_root.rglob("train"):
        if train_dir.is_dir() and (train_dir.parent / "test").is_dir():
            candidates.append(train_dir.parent)

    valid = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "train").is_dir() and (resolved / "test").is_dir():
            valid.append(resolved)

    if not valid:
        raise FileNotFoundError(
            "Could not find a folder containing train/ and test/ below "
            f"{input_root}."
        )

    return min(valid, key=lambda path: len(path.parts))


def infer_patient_id(filename: str) -> str:
    stem = Path(filename).stem
    for pattern in PATIENT_PATTERNS:
        match = pattern.match(stem)
        if match:
            return match.group(1)
    return stem


def infer_pneumonia_type(filename: str, class_name: str) -> str:
    if class_name.upper() == "NORMAL":
        return "normal"
    lowered = filename.lower()
    if "bacteria" in lowered:
        return "bacteria"
    if "virus" in lowered:
        return "virus"
    return "unknown"


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path: Path) -> tuple[int, int, str]:
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        mode = image.mode
    return width, height, mode


def build_manifest(
    input_root: Path,
    output_csv: Path,
    verify_images: bool = True,
    compute_hash: bool = False,
) -> pd.DataFrame:
    dataset_root = locate_dataset_root(input_root)
    records = []
    failures = []

    for source_split in SOURCE_SPLITS:
        split_dir = dataset_root / source_split
        if not split_dir.is_dir():
            continue

        for class_name, label in CLASS_TO_LABEL.items():
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")

            image_paths = sorted(
                path
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )

            for path in image_paths:
                width: Optional[int] = None
                height: Optional[int] = None
                mode: Optional[str] = None

                if verify_images:
                    try:
                        width, height, mode = inspect_image(path)
                    except Exception as exc:
                        failures.append((str(path), repr(exc)))
                        continue

                record = {
                    "path": path.relative_to(dataset_root).as_posix(),
                    "label": label,
                    "class_name": class_name,
                    "source_split": source_split,
                    "patient_id": infer_patient_id(path.name),
                    "pneumonia_type": infer_pneumonia_type(path.name, class_name),
                    "width": width,
                    "height": height,
                    "mode": mode,
                }
                if compute_hash:
                    record["sha256"] = file_sha256(path)
                records.append(record)

    if not records:
        raise RuntimeError(f"No supported images found below {dataset_root}")

    manifest = pd.DataFrame.from_records(records)
    output_csv = output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_csv, index=False)

    print(f"Dataset root: {dataset_root}")
    print(f"Manifest: {output_csv}")
    print(f"Valid images: {len(manifest)}")
    if failures:
        print(f"Skipped unreadable images: {len(failures)}")
        for path, error in failures[:5]:
            print(f"  {path}: {error}")

    print("Counts by source split and class:")
    print(
        manifest.groupby(["source_split", "class_name"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )

    if compute_hash:
        duplicate_rows = manifest[manifest.duplicated("sha256", keep=False)]
        if not duplicate_rows.empty:
            groups = duplicate_rows["sha256"].nunique()
            print(f"Warning: found {groups} duplicate-content groups.")

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a CXR image manifest CSV.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--no-verify-images",
        action="store_true",
        help="Skip opening every image during manifest creation.",
    )
    parser.add_argument(
        "--compute-hash",
        action="store_true",
        help="Compute SHA-256 values for duplicate/leakage checks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_manifest(
        input_root=args.input_root,
        output_csv=args.output,
        verify_images=not args.no_verify_images,
        compute_hash=args.compute_hash,
    )


if __name__ == "__main__":
    main()
