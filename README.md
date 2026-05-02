# Fruit YOLOv8n OpenCV + Streamlit

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate fruit-yolo-opencv
```

If the env already exists:

```bash
conda activate fruit-yolo-opencv
python -m pip install -r requirements.txt
```

Download YOLOv8n models:

```bash
python scripts/download_models.py --export-onnx
```

Prepare the fruit spoilage dataset:

```bash
python scripts/prepare_dataset.py
```

Train the spoilage classifier:

```bash
python scripts/train_spoilage_classifier.py --epochs 15 --device cpu
```

Run the Streamlit app:

```bash
streamlit run app.py
```

Run OpenCV webcam:

```bash
python scripts/webcam_detect.py --mode classify
python scripts/webcam_detect.py --mode detect
```

## Deployment Notes

- For Streamlit Cloud deployment, `opencv-python-headless` is used instead of `opencv-python` to avoid GUI library dependencies
- Environment variables are set in `app.py` to ensure headless operation
- The app includes graceful error handling if ultralytics fails to import
- The app uses YOLOv8n models for fruit detection and spoilage classification
- Models are loaded from the `models/` directory

Important: `dataset.zip` is an image classification dataset, not a YOLO bounding-box
detection dataset. Fresh/rotten prediction uses `yolov8n-cls.pt`; object boxes use the
pretrained COCO `yolov8n.pt`.

sudo apt update
sudo apt install -y python3-pip python3-smbus i2c-tools libgl1
python3 -m pip install RPi.GPIO Adafruit_DHT smbus2