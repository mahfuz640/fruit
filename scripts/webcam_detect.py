from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
DETECT_MODEL = ROOT / "models" / "yolov8n.pt"
CLASSIFY_MODEL = ROOT / "models" / "fruit_spoilage_yolov8n_cls.pt"
BASE_CLASSIFY_MODEL = ROOT / "models" / "yolov8n-cls.pt"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

from ultralytics import YOLO


def default_model(mode: str) -> Path:
    if mode == "detect":
        return DETECT_MODEL
    return CLASSIFY_MODEL if CLASSIFY_MODEL.exists() else BASE_CLASSIFY_MODEL


def classification_label(result) -> str:
    probs = result.probs
    class_id = int(probs.top1)
    confidence = float(probs.top1conf)
    return f"{result.names[class_id]} {confidence:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO webcam detection/classification with OpenCV.")
    parser.add_argument("--mode", choices=["detect", "classify"], default="classify")
    parser.add_argument("--model", default="", help="Override model path")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO image size")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else default_model(args.mode)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run python scripts/download_models.py first.")

    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    label = ""
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if args.mode == "detect":
            result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            frame = result.plot()
        else:
            if frame_index % 8 == 0:
                result = model.predict(frame, imgsz=224, verbose=False)[0]
                label = classification_label(result)
            cv2.putText(frame, label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 220, 80), 3)

        cv2.imshow("Fruit YOLOv8n", frame)
        frame_index += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
