import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, render_template, Response, request, jsonify
import threading
import base64
import time
import os

app = Flask(__name__)

# ─── Model Config ──────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "model", "best.onnx")
INPUT_SIZE  = 640
CONF_THRESH = 0.35
IOU_THRESH  = 0.45
CLASS_NAMES = ["pill"]

# ─── Load ONNX Model ───────────────────────────────────────────────────────────
session = None

def load_model():
    global session
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 2
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        MODEL_PATH,
        sess_options=opts,
        providers=["CPUExecutionProvider"]
    )
    print(f"[✓] Model loaded: {MODEL_PATH}")

# ─── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(image):
    img = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

# ─── Postprocessing ────────────────────────────────────────────────────────────
def postprocess(outputs, orig_shape):
    predictions = np.squeeze(outputs[0])
    if predictions.ndim == 1:
        predictions = predictions[np.newaxis, :]

    # YOLOv8: shape is (84, 8400) → transpose to (8400, 84)
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    orig_h, orig_w = orig_shape[:2]
    x_scale = orig_w / INPUT_SIZE
    y_scale  = orig_h / INPUT_SIZE

    boxes, scores = [], []
    for pred in predictions:
        class_scores = pred[4:]
        cls_id = int(np.argmax(class_scores))
        conf   = float(class_scores[cls_id])
        if conf < CONF_THRESH:
            continue

        cx, cy, w, h = pred[:4]
        x1 = int((cx - w / 2) * x_scale)
        y1 = int((cy - h / 2) * y_scale)
        x2 = int((cx + w / 2) * x_scale)
        y2 = int((cy + h / 2) * y_scale)

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(conf)

    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, IOU_THRESH)
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            results.append({
                "box":   [x, y, x + w, y + h],
                "score": round(scores[i], 3)
            })
    return results

# ─── Inference ─────────────────────────────────────────────────────────────────
def run_inference(image):
    blob   = preprocess(image)
    inputs = {session.get_inputs()[0].name: blob}
    outs   = session.run(None, inputs)
    return postprocess(outs, image.shape)

# ─── Draw Detections ───────────────────────────────────────────────────────────
def draw_detections(image, detections):
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        score = det["score"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 120), 2)
        label = f"pill {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 220, 120), -1)
        cv2.putText(image, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    count = len(detections)
    cv2.putText(image, f"Pills: {count}", (14, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, f"Pills: {count}", (14, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 220, 120), 2, cv2.LINE_AA)
    return image

# ─── Avg Confidence Helper ─────────────────────────────────────────────────────
def avg_confidence(detections):
    if not detections:
        return None
    return round(sum(d["score"] for d in detections) / len(detections), 3)

# ─── Camera Stream ─────────────────────────────────────────────────────────────
camera_lock   = threading.Lock()
active_cam    = None
cam_index     = 0
stream_active = False

def open_camera(index=0):
    global active_cam, cam_index
    with camera_lock:
        if active_cam is not None:
            active_cam.release()
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 20)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            return False
        active_cam = cap
        cam_index  = index
        return True

def generate_frames():
    global stream_active
    stream_active = True
    frame_interval = 1 / 15
    last_time = 0

    try:
        while stream_active:
            now = time.time()
            if now - last_time < frame_interval:
                time.sleep(0.01)
                continue
            last_time = now

            with camera_lock:
                if active_cam is None or not active_cam.isOpened():
                    break
                ret, frame = active_cam.read()

            if not ret:
                time.sleep(0.05)
                continue

            detections = run_inference(frame)
            frame      = draw_detections(frame, detections)
            _, buffer  = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buffer.tobytes() + b"\r\n")
    finally:
        stream_active = False

# ─── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/live")
def live():
    return render_template("live.html")

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed_snap")
def video_feed_snap():
    """Single frame JSON — returns pill count + avg confidence for sidebar polling."""
    with camera_lock:
        if active_cam is None or not active_cam.isOpened():
            return jsonify({"count": 0, "avg_conf": None})
        ret, frame = active_cam.read()
    if not ret:
        return jsonify({"count": 0, "avg_conf": None})
    detections = run_inference(frame)
    return jsonify({
        "count":    len(detections),
        "avg_conf": avg_confidence(detections)
    })

@app.route("/start_camera", methods=["POST"])
def start_camera():
    idx = int(request.json.get("index", 0))
    ok  = open_camera(idx)
    return jsonify({"success": ok, "index": idx})

@app.route("/stop_camera", methods=["POST"])
def stop_camera():
    global stream_active, active_cam
    stream_active = False
    time.sleep(0.2)
    with camera_lock:
        if active_cam:
            active_cam.release()
            active_cam = None
    return jsonify({"success": True})

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file  = request.files["image"]
    data  = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    detections = run_inference(image)
    result_img = draw_detections(image.copy(), detections)
    _, buf = cv2.imencode(".jpg", result_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64    = base64.b64encode(buf).decode("utf-8")

    return jsonify({
        "count":      len(detections),
        "avg_conf":   avg_confidence(detections),
        "detections": detections,
        "image":      f"data:image/jpeg;base64,{b64}"
    })

# ─── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_model()
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)