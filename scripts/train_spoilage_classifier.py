from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "fruit_spoilage_yolo_cls"
DEFAULT_BASE_MODEL = ROOT / "models" / "yolov8n-cls.pt"
DEFAULT_OUTPUT = ROOT / "models" / "fruit_spoilage_yolov8n_cls.pt"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8n classification model on fruit spoilage data.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Prepared YOLO classification dataset")
    parser.add_argument("--model", default=str(DEFAULT_BASE_MODEL), help="Base YOLOv8n classification model")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=224, help="Image size")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--device", default="", help="Example: 0 for GPU, cpu for CPU")
    parser.add_argument("--name", default="fruit_spoilage_yolov8n_cls", help="Run name")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Where to save best model")
    args = parser.parse_args()

    data_dir = Path(args.data)
    model_path = Path(args.model)
    output_path = Path(args.output)

    if not (data_dir / "train").exists() or not (data_dir / "val").exists():
        raise FileNotFoundError(
            f"Prepared data not found at {data_dir}. Run: python scripts/prepare_dataset.py"
        )

    model = YOLO(str(model_path) if model_path.exists() else "yolov8n-cls.pt")
    train_args = {
        "data": str(data_dir),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(ROOT / "runs" / "classify"),
        "name": args.name,
    }
    if args.device:
        train_args["device"] = args.device

    results = model.train(**train_args)
    save_dir = Path(results.save_dir)
    best_path = save_dir / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Training finished but best.pt was not found: {best_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, output_path)
    print(f"Saved trained model: {output_path}")


if __name__ == "__main__":
    main()
