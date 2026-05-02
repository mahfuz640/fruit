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

Important: `dataset.zip` is an image classification dataset, not a YOLO bounding-box
detection dataset. Fresh/rotten prediction uses `yolov8n-cls.pt`; object boxes use the
pretrained COCO `yolov8n.pt`.

sudo apt update
sudo apt install -y python3-pip python3-smbus i2c-tools
python3 -m pip install RPi.GPIO Adafruit_DHT smbus2