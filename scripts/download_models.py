from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

from ultralytics import YOLO


def save_model(model_name: str, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        print(f"Already saved: {target_path}")
        return target_path

    model = YOLO(model_name)
    source = Path(getattr(model, "ckpt_path", ""))
    if not source.exists():
        source = Path(model_name)
    if not source.exists():
        raise FileNotFoundError(f"Downloaded model file not found for {model_name}")

    shutil.copy2(source, target_path)
    print(f"Saved: {target_path}")
    return target_path


def export_onnx(pt_path: Path, onnx_path: Path) -> None:
    if onnx_path.exists():
        print(f"Already exported: {onnx_path}")
        return

    model = YOLO(str(pt_path))
    exported = Path(model.export(format="onnx", imgsz=640, opset=12, simplify=False))
    if exported.exists() and exported.resolve() != onnx_path.resolve():
        shutil.copy2(exported, onnx_path)
    print(f"Exported: {onnx_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download YOLOv8n models for detection and classification.")
    parser.add_argument("--export-onnx", action="store_true", help="Also export yolov8n.pt to ONNX for OpenCV DNN")
    args = parser.parse_args()

    detector = save_model("yolov8n.pt", MODEL_DIR / "yolov8n.pt")
    save_model("yolov8n-cls.pt", MODEL_DIR / "yolov8n-cls.pt")

    if args.export_onnx:
        export_onnx(detector, MODEL_DIR / "yolov8n.onnx")


if __name__ == "__main__":
    main()
