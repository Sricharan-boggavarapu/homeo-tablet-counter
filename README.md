# PillDetect — AI Pill Counter

Real-time pill detection and counting using a trained YOLOv8 ONNX model.
Runs entirely on CPU via Flask + ONNX Runtime.

---

## Project Structure

```
pill_counter_final/
├── app.py                  ← Flask app + inference + streaming
├── requirements.txt
├── README.md
├── model/
│   └── best.onnx           ← Place your model here  ← IMPORTANT
├── utils/
│   ├── __init__.py
│   └── inference.py        ← Standalone inference helpers
├── templates/
│   ├── index.html          ← Home page
│   ├── live.html           ← Live camera feed
│   └── upload.html         ← Image upload + detection
└── static/
    └── style.css
```

---

## Setup Instructions

### 1. Place your model

Copy `best.onnx` from:
```
pill_model_run/weights/best.onnx
```
into:
```
pill_counter_final/model/best.onnx
```

### 2. Create a virtual environment (recommended)

```bash
cd pill_counter_final
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `flask` — web server
- `opencv-python` — camera access + image processing
- `onnxruntime` — ONNX model inference on CPU
- `numpy` — array operations

### 4. Run the app

```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## Features

| Feature | Details |
|---|---|
| Live camera stream | MJPEG stream at ~15 FPS |
| Camera selection | Index 0, 1, 2 (dropdown in UI) |
| Bounding boxes | Green boxes with confidence scores |
| Pill count overlay | Drawn directly on frame |
| Sidebar count | Polled every 800ms via `/video_feed_snap` |
| Image upload | Drag-and-drop or file picker |
| Upload results | Side-by-side original vs detected |
| Per-detection list | Confidence score per pill |

---

## API Routes

| Route | Method | Description |
|---|---|---|
| `/` | GET | Home page |
| `/live` | GET | Live camera page |
| `/upload` | GET | Image upload page |
| `/video_feed` | GET | MJPEG stream |
| `/video_feed_snap` | GET | JSON with current pill count |
| `/start_camera` | POST | Open camera `{"index": 0}` |
| `/stop_camera` | POST | Release camera |
| `/detect` | POST | Run inference on uploaded image |

---

## Configuration (app.py top section)

```python
INPUT_SIZE  = 640    # model input resolution
CONF_THRESH = 0.35   # detection confidence threshold
IOU_THRESH  = 0.45   # NMS IoU threshold
```

Lower `CONF_THRESH` → more detections (more false positives).
Raise `CONF_THRESH` → fewer but more certain detections.

---

## Troubleshooting

**Camera not opening**
- Try changing the camera index (0, 1, 2) in the dropdown
- On Windows, make sure no other app is using the camera

**Model not loading**
- Confirm `model/best.onnx` exists
- Check the path in `app.py` → `MODEL_PATH`

**Slow inference**
- Normal on CPU — YOLOv8n/s is recommended for real-time
- Frames are resized to 640×640 then processed

**Port already in use**
```bash
# Change port in app.py
app.run(port=5001)
```
