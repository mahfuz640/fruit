from pathlib import Path
import argparse

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "yolov8n.onnx"

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def letterbox(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, tuple[int, int]]:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width, new_height = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - new_width) // 2
    pad_y = (size - new_height) // 2
    canvas[pad_y : pad_y + new_height, pad_x : pad_x + new_width] = resized
    return canvas, scale, (pad_x, pad_y)


def detect(
    image: np.ndarray,
    net: cv2.dnn.Net,
    confidence_threshold: float,
    nms_threshold: float,
    size: int = 640,
) -> list[tuple[int, float, tuple[int, int, int, int]]]:
    blob_image, scale, (pad_x, pad_y) = letterbox(image, size)
    blob = cv2.dnn.blobFromImage(blob_image, 1 / 255.0, (size, size), swapRB=True, crop=False)
    net.setInput(blob)
    output = net.forward()

    predictions = np.squeeze(output)
    if predictions.ndim == 2 and predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    boxes, scores, class_ids = [], [], []
    original_height, original_width = image.shape[:2]

    for row in predictions:
        class_scores = row[4:]
        class_id = int(np.argmax(class_scores))
        score = float(class_scores[class_id])
        if score < confidence_threshold:
            continue

        center_x, center_y, width, height = row[:4]
        x1 = int((center_x - width / 2 - pad_x) / scale)
        y1 = int((center_y - height / 2 - pad_y) / scale)
        x2 = int((center_x + width / 2 - pad_x) / scale)
        y2 = int((center_y + height / 2 - pad_y) / scale)

        x1 = max(0, min(x1, original_width - 1))
        y1 = max(0, min(y1, original_height - 1))
        x2 = max(0, min(x2, original_width - 1))
        y2 = max(0, min(y2, original_height - 1))

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(score)
        class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, scores, confidence_threshold, nms_threshold)
    if len(indexes) == 0:
        return []

    return [(class_ids[i], scores[i], tuple(boxes[i])) for i in np.array(indexes).flatten()]


def draw_detections(
    image: np.ndarray,
    detections: list[tuple[int, float, tuple[int, int, int, int]]],
) -> np.ndarray:
    result = image.copy()
    for class_id, score, (x, y, width, height) in detections:
        label = COCO_NAMES[class_id] if class_id < len(COCO_NAMES) else str(class_id)
        cv2.rectangle(result, (x, y), (x + width, y + height), (0, 190, 80), 2)
        cv2.putText(
            result,
            f"{label} {score:.2f}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 190, 80),
            2,
            cv2.LINE_AA,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLOv8n ONNX with OpenCV DNN.")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="YOLOv8n ONNX model path")
    parser.add_argument("--output", default="outputs/detection.jpg", help="Output image path")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS threshold")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    net = cv2.dnn.readNetFromONNX(args.model)
    detections = detect(image, net, args.conf, args.nms)
    result = draw_detections(image, detections)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)

    print(f"Detections: {len(detections)}")
    print(f"Saved output: {output_path}")


if __name__ == "__main__":
    main()
