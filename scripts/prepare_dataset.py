from __future__ import annotations

import argparse
import os
import random
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP = ROOT / "dataset.zip"
DEFAULT_RAW = ROOT / "data" / "raw"
DEFAULT_OUT = ROOT / "data" / "fruit_spoilage_yolo_cls"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}

CLASS_FIXES = {
    "freshpatato": "freshpotato",
    "rottenpatato": "rottenpotato",
    "freshtamto": "freshtomato",
    "rottentamto": "rottentomato",
    "freshbittergroud": "freshbittergourd",
    "rottenbittergroud": "rottenbittergourd",
}


def canonical_class(name: str) -> str:
    return CLASS_FIXES.get(name.lower(), name.lower())


def extract_dataset(zip_path: Path, raw_dir: Path) -> None:
    train_dir = raw_dir / "Train"
    test_dir = raw_dir / "Test"
    if train_dir.exists() and test_dir.exists():
        print(f"Raw dataset already exists: {raw_dir}")
        return

    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} -> {raw_dir}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(raw_dir)


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(source.resolve(), destination)
    except OSError:
        shutil.copy2(source, destination)


def add_file(source: Path, split: str, class_name: str, out_dir: Path) -> None:
    safe_name = f"{source.parent.name}__{source.name}"
    destination = out_dir / split / class_name / safe_name
    link_or_copy(source, destination)


def prepare_train_val(raw_dir: Path, out_dir: Path, val_ratio: float, seed: int) -> None:
    rng = random.Random(seed)
    train_root = raw_dir / "Train"

    for class_dir in sorted(path for path in train_root.iterdir() if path.is_dir()):
        class_name = canonical_class(class_dir.name)
        files = image_files(class_dir)
        rng.shuffle(files)

        val_count = max(1, round(len(files) * val_ratio)) if len(files) > 1 else 0
        val_files = set(files[:val_count])

        for source in files:
            split = "val" if source in val_files else "train"
            add_file(source, split, class_name, out_dir)


def prepare_test(raw_dir: Path, out_dir: Path) -> None:
    test_root = raw_dir / "Test"
    if not test_root.exists():
        return

    for class_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
        class_name = canonical_class(class_dir.name)
        for source in image_files(class_dir):
            add_file(source, "test", class_name, out_dir)


def count_images(out_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        split_dir = out_dir / split
        counts[split] = sum(1 for path in split_dir.rglob("*") if path.is_file()) if split_dir.exists() else 0
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare fruit spoilage data for YOLOv8 classification.")
    parser.add_argument("--zip", default=str(DEFAULT_ZIP), help="Path to dataset.zip")
    parser.add_argument("--raw", default=str(DEFAULT_RAW), help="Raw extracted dataset directory")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Prepared YOLO classification directory")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split from Train")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    raw_dir = Path(args.raw)
    out_dir = Path(args.out)

    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset zip not found: {zip_path}")

    extract_dataset(zip_path, raw_dir)
    prepare_train_val(raw_dir, out_dir, args.val_ratio, args.seed)
    prepare_test(raw_dir, out_dir)

    print(f"Prepared dataset: {out_dir}")
    for split, count in count_images(out_dir).items():
        print(f"{split}: {count} images")


if __name__ == "__main__":
    main()
