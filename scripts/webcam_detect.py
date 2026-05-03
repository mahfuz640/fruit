from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
DETECT_MODEL = ROOT / "models" / "yolov8n.pt"
CLASSIFY_MODEL = ROOT / "models" / "fruit_spoilage_yolov8n_cls.pt"
ULTRALYTICS_CONFIG_DIR = ROOT / ".ultralytics"
ULTRALYTICS_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))

from ultralytics import YOLO

FRESH_MARKERS = ("fresh",)
SPOILED_MARKERS = ("rotten", "spoiled", "mold", "mould", "decay", "bad")
FRUIT_NAME_FIXES = {
    "apple": "Apple",
    "banana": "Banana",
    "bittergourd": "Bitter Gourd",
    "carrot": "Carrot",
    "cucumber": "Cucumber",
    "mango": "Mango",
    "orange": "Orange",
    "potato": "Potato",
    "tomato": "Tomato",
}


def default_model(mode: str) -> Path:
    if mode == "detect":
        return DETECT_MODEL
    return CLASSIFY_MODEL


def infer_freshness(class_name: str) -> str:
    normalized = class_name.lower().replace("_", "").replace("-", "").replace(" ", "")
    if any(marker in normalized for marker in SPOILED_MARKERS):
        return "Spoiled"
    if any(marker in normalized for marker in FRESH_MARKERS):
        return "Fresh"
    return "Unknown"


def infer_fruit_name(class_name: str) -> str:
    normalized = class_name.lower().replace("_", "").replace("-", "").replace(" ", "")
    for marker in (*SPOILED_MARKERS, *FRESH_MARKERS):
        normalized = normalized.replace(marker, "")
    return FRUIT_NAME_FIXES.get(normalized, class_name.replace("_", " ").replace("-", " ").title())


def classification_label(result) -> str:
    probs = result.probs
    class_id = int(probs.top1)
    confidence = float(probs.top1conf)
    class_name = str(result.names[class_id])
    freshness = infer_freshness(class_name)
    fruit = infer_fruit_name(class_name)
    if freshness == "Unknown":
        return f"{fruit} {confidence:.2f}"
    return f"{freshness}: {fruit} {confidence:.2f}"


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
        if args.mode == "classify":
            raise FileNotFoundError(
                f"Spoilage classifier model not found: {model_path}. "
                "Run python scripts/prepare_dataset.py, then "
                "python scripts/train_spoilage_classifier.py --epochs 15 --device cpu."
            )
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
