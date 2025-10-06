import os
import time
import uuid
import json
import shutil
import logging
from typing import Dict, List

import cv2
import requests
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from ultralytics import YOLO
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ppe-frame-server")

# Environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "model/best.pt")
OJITOO_BASE_URL = os.getenv("OJITOO_BASE_URL", "").rstrip("/")
AUTH_TOKEN = os.getenv("AUTHORIZATION_TOKEN", "")
YOLO_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.4"))
ALERT_COOLDOWN_SECONDS = float(os.getenv("ALERT_COOLDOWN_SECONDS", "30"))

# Derived constants
ALERT_ENDPOINT = f"{OJITOO_BASE_URL}/alerts/"

# Model loading
logger.info(f"Loading YOLO model from {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# Track last alert timestamps
last_alert_time: Dict[str, float] = {}

# FastAPI app
app = FastAPI(title="PPE Detection API", version="1.0")

# ---------------------------------------------------------------------
# PPE Mappings and Colors
# ---------------------------------------------------------------------
CLASS_TO_PPE = {
        "no-hardhat": "hardhat",
        "no-gloves": "gloves",
        "no-vest": "vest",
        "no-mask": "mask",
        "no-goggles": "goggles",
        "no-earplugs": "earplugs",
        }

PPE_COLORS = {
        "hardhat": (0, 0, 255),      # Red
        "gloves": (0, 255, 0),       # Green
        "vest": (255, 0, 0),         # Blue
        "mask": (0, 255, 255),       # Yellow
        "goggles": (255, 0, 255),    # Magenta
        "earplugs": (255, 165, 0),   # Orange
        }


# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------
def save_temp_file(upload: UploadFile) -> str:
    """Save uploaded file to a temporary location."""
    temp_path = f"temp_{uuid.uuid4()}.jpg"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)
    return temp_path


def draw_bounding_boxes(image_path: str, detections: List[Dict]) -> str:
    """Draw bounding boxes and labels on the image."""
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Cannot open image: {image_path}")
        return image_path

    for det in detections:
        ppe_type = det["missingPpe"]
        bbox = det["bbox"]
        conf = det["confidence"]
        color = PPE_COLORS.get(ppe_type, (255, 255, 255))

        x1, y1, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        x2, y2 = x1 + w, y1 + h

        # Draw rectangle and label
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"No {ppe_type}: {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - th - baseline - 4), (x1 + tw, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    annotated_path = f"annotated_{uuid.uuid4()}.jpg"
    cv2.imwrite(annotated_path, image)
    return annotated_path


def build_alert_payload(camera_id: str, detections: List[Dict]) -> Dict:
    """Constructs the alert payload."""
    return {
            "cameraId": camera_id,
            "status": "new",
            "ppeDetections": detections,
            }


def send_alert(payload: Dict, image_path: str) -> bool:
    """Send alert with annotated image to backend."""
    headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
    try:
        with open(image_path, "rb") as img_file:
            files = {"file": ("frame.jpg", img_file, "image/jpeg")}
            data = {"payload": json.dumps(payload)}
            response = requests.post(ALERT_ENDPOINT, data=data, files=files,
                                     headers=headers, timeout=15)
            response.raise_for_status()
            logger.info(f"Alert sent successfully [{response.status_code}]")
            return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send alert: {e}")
        return False


def detect_missing_ppe(image_path: str) -> List[Dict]:
    """Run YOLO model on the image and return detections for missing PPE."""
    results = model.predict(image_path, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)
    detections = []

    for r in results:
        for box in r.boxes:
            class_name = model.names[int(box.cls)]
            conf = float(box.conf)
            if class_name not in CLASS_TO_PPE:
                continue

            if conf >= YOLO_CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "missingPpe": CLASS_TO_PPE[class_name],
                    "confidence": conf,
                    "bbox": {
                        "x": x1,
                        "y": y1,
                        "width": x2 - x1,
                        "height": y2 - y1
                        }
                    })
    return detections


def handle_alert(camera_id: str, detections: List[Dict], annotated_image: str) -> bool:
    """Handle cooldown logic and alert sending."""
    now = time.time()
    last_time = last_alert_time.get(camera_id, 0)
    if now - last_time < ALERT_COOLDOWN_SECONDS:
        logger.info(f"⏳ Skipping alert for {camera_id} (cooldown active)")
        return False

    payload = build_alert_payload(camera_id, detections)
    success = send_alert(payload, annotated_image)
    if success:
        last_alert_time[camera_id] = now
    return success


# ---------------------------------------------------------------------
# API Endpoint
# ---------------------------------------------------------------------
@app.post("/analyze/")
async def analyze_frame(file: UploadFile = File(...), cameraId: str = Form(...)):
    """
    Process a single camera frame, detect missing PPE, and send alert if needed.
    """
    temp_file = save_temp_file(file)
    detections = detect_missing_ppe(temp_file)
    annotated_path = None
    alert_sent = False

    if detections:
        annotated_path = draw_bounding_boxes(temp_file, detections)
        alert_sent = handle_alert(cameraId, detections, annotated_path)

    # Cleanup
    for path in [temp_file, annotated_path]:
        if path and os.path.exists(path):
            os.remove(path)

    return {
            "cameraId": cameraId,
            "detections": detections,
            "alertSent": alert_sent,
            }

