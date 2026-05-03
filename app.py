from __future__ import annotations

import os
from pathlib import Path

# Set environment variables for headless OpenCV before any imports
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", "/usr/lib/x86_64-linux-gnu/qt5/plugins")
os.environ.setdefault("DISPLAY", "")
os.environ.setdefault("OPENCV_HEADLESS", "1")
os.environ.setdefault("LIBGL_ALWAYS_INDIRECT", "1")
os.environ.setdefault("LIBVA_DRIVER_NAME", "i965")
os.environ.setdefault("LIBGL_DRIVERS_PATH", "/usr/lib/x86_64-linux-gnu/dri")

ROOT = Path(__file__).resolve().parent
ULTRALYTICS_CONFIG_DIR = ROOT / ".ultralytics"
ULTRALYTICS_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))

import numpy as np
import streamlit as st
import cv2
from PIL import Image

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import ultralytics: {e}")
    st.error("Please check that all dependencies are installed correctly.")
    ULTRALYTICS_AVAILABLE = False
    YOLO = None

try:
    import av
    from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
    WEBRTC_IMPORT_ERROR = None
except Exception as exc:
    av = None
    VideoProcessorBase = object
    webrtc_streamer = None
    WEBRTC_IMPORT_ERROR = exc


MODELS = ROOT / "models"
DETECT_MODEL = MODELS / "yolov8n.pt"
CLASSIFY_MODEL = MODELS / "fruit_spoilage_yolov8n_cls.pt"
BASE_CLASSIFY_MODEL = MODELS / "yolov8n-cls.pt"
CLASSIFY_EVERY_N_FRAMES = 8

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


def display_class_name(class_name: str) -> str:
    return class_name.replace("_", " ").replace("-", " ").strip().title()


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
    if normalized in FRUIT_NAME_FIXES:
        return FRUIT_NAME_FIXES[normalized]
    return display_class_name(class_name)


def classification_summary(result) -> dict[str, str | float]:
    probs = result.probs
    class_id = int(probs.top1)
    class_name = str(result.names[class_id])
    confidence = float(probs.top1conf)
    return {
        "class_name": class_name,
        "display_class": display_class_name(class_name),
        "freshness": infer_freshness(class_name),
        "fruit": infer_fruit_name(class_name),
        "confidence": confidence,
    }


def format_live_classification(summary: dict[str, str | float]) -> tuple[str, tuple[int, int, int]]:
    confidence = float(summary["confidence"])
    freshness = str(summary["freshness"])
    fruit = str(summary["fruit"])
    if freshness == "Fresh":
        color = (40, 180, 60)
        label = f"Fresh: {fruit} ({confidence:.0%})"
    elif freshness == "Spoiled":
        color = (40, 40, 220)
        label = f"Spoiled: {fruit} ({confidence:.0%})"
    else:
        color = (40, 150, 220)
        label = f"{summary['display_class']} ({confidence:.0%})"
    return label, color


def draw_label(frame: np.ndarray, label: str, color: tuple[int, int, int]) -> np.ndarray:
    cv2.rectangle(frame, (14, 16), (min(frame.shape[1] - 14, 620), 70), (25, 25, 25), -1)
    cv2.putText(frame, label, (24, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return frame


def render_about() -> None:
    st.subheader("About")
    st.markdown(
        """
        This application uses YOLOv8n to detect fruits and, after training the spoilage
        classifier, predicts whether the fruit is **Fresh** or **Spoiled**.

        **How to use**

        1. Select **Spoilage classifier** to check Fresh/Spoiled status, or **Fruit detector** to draw fruit/object boxes.
        2. Use **Camera / upload** to take a photo or upload an image.
        3. Use **Live camera** for real-time prediction from the browser camera.

        **Developer**

        Created by **Engr. Md. Mahfuzur Rahman**  
        **MSc in IIT at Jahangirnagar University**
        """
    )


@st.cache_resource
def load_model(path: str):
    if not ULTRALYTICS_AVAILABLE:
        return None
    return YOLO(path)


def classify_image(image: Image.Image, model_path: Path) -> None:
    if not ULTRALYTICS_AVAILABLE:
        st.error("YOLO model not available. Please check dependencies.")
        return
    
    model = load_model(str(model_path))
    if model is None:
        st.error("Failed to load model.")
        return
        
    result = model.predict(np.array(image), imgsz=224, verbose=False)[0]
    probs = result.probs
    summary = classification_summary(result)

    metric_freshness, metric_fruit, metric_class = st.columns(3)
    metric_freshness.metric(
        "Freshness",
        str(summary["freshness"]),
        f"{float(summary['confidence']):.1%}",
    )
    metric_fruit.metric("Fruit", str(summary["fruit"]))
    metric_class.metric("Model class", str(summary["display_class"]))

    if summary["freshness"] == "Unknown":
        st.warning(
            "This model class does not contain fresh/spoiled information. "
            "Train the spoilage classifier to get Fresh or Spoiled predictions."
        )

    rows = [
        {
            "fruit": infer_fruit_name(str(result.names[int(class_id)])),
            "freshness": infer_freshness(str(result.names[int(class_id)])),
            "model_class": display_class_name(str(result.names[int(class_id)])),
            "confidence": float(probs.data[int(class_id)]),
        }
        for class_id in probs.top5
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)


def detect_image(image: Image.Image, model_path: Path, confidence: float) -> None:
    if not ULTRALYTICS_AVAILABLE:
        st.error("YOLO model not available. Please check dependencies.")
        return
        
    model = load_model(str(model_path))
    if model is None:
        st.error("Failed to load model.")
        return
        
    result = model.predict(np.array(image), conf=confidence, imgsz=640, verbose=False)[0]
    st.image(result.plot(), channels="BGR", use_container_width=True)


class FruitVideoProcessor(VideoProcessorBase):
    def __init__(self, mode: str, model_path: str, confidence: float) -> None:
        if not ULTRALYTICS_AVAILABLE:
            self.model = None
        else:
            self.model = YOLO(model_path)
        self.mode = mode
        self.confidence = confidence
        self.frame_index = 0
        self.classification_label = "Point the camera at a fruit"
        self.classification_color = (40, 150, 220)

    def recv(self, frame):
        if self.model is None:
            return frame  # Return original frame if model not available
        image = frame.to_ndarray(format="bgr24")

        if self.mode == "detect":
            result = self.model.predict(image, conf=self.confidence, imgsz=640, verbose=False)[0]
            annotated = result.plot()
        else:
            if self.frame_index % CLASSIFY_EVERY_N_FRAMES == 0:
                result = self.model.predict(image, imgsz=224, verbose=False)[0]
                self.classification_label, self.classification_color = format_live_classification(
                    classification_summary(result)
                )
            annotated = draw_label(image, self.classification_label, self.classification_color)
            self.frame_index += 1

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


def main() -> None:
    st.set_page_config(page_title="Fruit YOLOv8n", layout="wide")
    st.title("Fruit YOLOv8n")

    if not ULTRALYTICS_AVAILABLE:
        st.error("YOLO dependencies not available. The app cannot function without ultralytics.")
        st.stop()

    mode = st.sidebar.radio("Mode", ["Spoilage classifier", "Fruit detector"], index=1)
    confidence = st.sidebar.slider("Confidence", 0.10, 0.90, 0.35, 0.05)

    model_path = CLASSIFY_MODEL if mode == "Spoilage classifier" else DETECT_MODEL
    model_missing = not model_path.exists()

    if model_missing:
        if mode == "Spoilage classifier":
            st.error(
                "Spoilage classifier model not found. Train it first so predictions can say Fresh or Spoiled."
            )
            st.code(
                "python scripts/prepare_dataset.py\n"
                "python scripts/train_spoilage_classifier.py --epochs 15 --device cpu",
                language="bash",
            )
        else:
            st.error(f"Model not found: {model_path}. Run: python scripts/download_models.py")

    tab_about, tab_snapshot, tab_live = st.tabs(["About", "Camera / upload", "Live camera"])

    with tab_about:
        render_about()

    with tab_snapshot:
        source = st.camera_input("Camera")
        uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"])
        image_source = uploaded or source

        if image_source is not None:
            if model_missing:
                st.warning("Add the selected model first, then try prediction again.")
            else:
                image = Image.open(image_source).convert("RGB")
                st.image(image, use_container_width=True)
                if mode == "Spoilage classifier":
                    classify_image(image, model_path)
                else:
                    detect_image(image, model_path, confidence)

    with tab_live:
        live_model_missing = not DETECT_MODEL.exists()
        if live_model_missing:
            st.error(f"Model not found: {DETECT_MODEL}. Run: python scripts/download_models.py")
        elif webrtc_streamer is None:
            st.error(f"Live camera dependency error: {WEBRTC_IMPORT_ERROR}")
            st.code("python -m pip install -r requirements.txt", language="bash")
        else:
            webrtc_streamer(
                key=f"fruit-live-detector-{DETECT_MODEL.name}",
                video_processor_factory=lambda: FruitVideoProcessor(
                    "detect",
                    str(DETECT_MODEL),
                    confidence,
                ),
                media_stream_constraints={
                    "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
                    "audio": False,
                },
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                desired_playing_state=True,
                async_processing=True,
            )


if __name__ == "__main__":
    main()
