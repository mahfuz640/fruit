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
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

import numpy as np
import streamlit as st
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
except Exception:
    av = None
    VideoProcessorBase = object
    webrtc_streamer = None


MODELS = ROOT / "models"
DETECT_MODEL = MODELS / "yolov8n.pt"
CLASSIFY_MODEL = MODELS / "fruit_spoilage_yolov8n_cls.pt"
BASE_CLASSIFY_MODEL = MODELS / "yolov8n-cls.pt"


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
    top1 = int(probs.top1)
    confidence = float(probs.top1conf)

    st.metric("Prediction", result.names[top1], f"{confidence:.1%}")
    rows = [
        {"class": result.names[int(class_id)], "confidence": float(probs.data[int(class_id)])}
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


class DetectorVideoProcessor(VideoProcessorBase):
    def __init__(self, model_path: str, confidence: float) -> None:
        if not ULTRALYTICS_AVAILABLE:
            self.model = None
        else:
            self.model = YOLO(model_path)
        self.confidence = confidence

    def recv(self, frame):
        if self.model is None:
            return frame  # Return original frame if model not available
        image = frame.to_ndarray(format="bgr24")
        result = self.model.predict(image, conf=self.confidence, imgsz=640, verbose=False)[0]
        annotated = result.plot()
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


def main() -> None:
    st.set_page_config(page_title="Fruit YOLOv8n", layout="wide")
    st.title("Fruit YOLOv8n")

    if not ULTRALYTICS_AVAILABLE:
        st.error("YOLO dependencies not available. The app cannot function without ultralytics.")
        st.stop()

    mode = st.sidebar.radio("Mode", ["Spoilage classifier", "Fruit detector"])
    confidence = st.sidebar.slider("Confidence", 0.10, 0.90, 0.35, 0.05)

    if mode == "Spoilage classifier":
        model_path = CLASSIFY_MODEL if CLASSIFY_MODEL.exists() else BASE_CLASSIFY_MODEL
        if not CLASSIFY_MODEL.exists():
            st.warning("Train the spoilage model first: python scripts/train_spoilage_classifier.py")
    else:
        model_path = DETECT_MODEL

    if not model_path.exists():
        st.error(f"Model not found: {model_path}. Run: python scripts/download_models.py")
        return

    tab_snapshot, tab_live = st.tabs(["Camera / upload", "Live detector"])

    with tab_snapshot:
        source = st.camera_input("Camera")
        uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"])
        image_source = uploaded or source

        if image_source is not None:
            image = Image.open(image_source).convert("RGB")
            st.image(image, use_container_width=True)
            if mode == "Spoilage classifier":
                classify_image(image, model_path)
            else:
                detect_image(image, model_path, confidence)

    with tab_live:
        if mode != "Fruit detector":
            st.info("Live mode uses yolov8n object detection. Use Camera / upload for spoilage classification.")
        elif webrtc_streamer is None:
            st.info("streamlit-webrtc is not installed.")
        else:
            webrtc_streamer(
                key="fruit-detector",
                video_processor_factory=lambda: DetectorVideoProcessor(str(DETECT_MODEL), confidence),
                media_stream_constraints={"video": True, "audio": False},
            )


if __name__ == "__main__":
    main()
