"""
utils/inference.py
Standalone inference helpers — can be imported independently for testing.
"""
import cv2
import numpy as np

INPUT_SIZE  = 640
CONF_THRESH = 0.35
IOU_THRESH  = 0.45


def preprocess(image: np.ndarray) -> np.ndarray:
    """BGR image → ONNX input blob (1, 3, 640, 640) float32."""
    img = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)


def postprocess(outputs, orig_shape) -> list[dict]:
    """Parse raw ONNX outputs → list of {box, score} dicts."""
    predictions = np.squeeze(outputs[0])
    if predictions.ndim == 1:
        predictions = predictions[np.newaxis, :]
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T          # → (8400, 84)

    orig_h, orig_w = orig_shape[:2]
    x_scale, y_scale = orig_w / INPUT_SIZE, orig_h / INPUT_SIZE

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
        boxes.append([x1, y1, int(w * x_scale), int(h * y_scale)])
        scores.append(conf)

    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, IOU_THRESH)
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            results.append({"box": [x, y, x + w, y + h], "score": round(scores[i], 3)})
    return results


def draw_detections(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw bounding boxes + pill count on image (in-place copy)."""
    img = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        score = det["score"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 120), 2)
        label = f"pill {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 220, 120), -1)
        cv2.putText(img, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    count = len(detections)
    cv2.putText(img, f"Pills: {count}", (14, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, f"Pills: {count}", (14, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 220, 120), 2, cv2.LINE_AA)
    return img
