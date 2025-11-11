import os
import time
import uuid
import json
import shutil
import logging
from datetime import datetime, timezone
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
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.3"))
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
CLASS_COLORS = {
        "person": (255, 255, 255),          # White
        "head": (200, 200, 200),            # Light Gray
        "face": (255, 220, 185),            # Skin tone
        "glasses": (255, 0, 255),           # Magenta
        "face-mask-medical": (0, 255, 255), # Cyan
        "face-guard": (0, 128, 255),        # Light Blue
        "ear": (210, 180, 140),             # Tan
        "ear-mufs": (255, 165, 0),          # Orange
        "hands": (255, 200, 150),           # Light Skin
        "gloves": (0, 255, 0),              # Green
        "foot": (150, 75, 0),               # Brown
        "shoes": (0, 0, 255),               # Blue
        "safety-vest": (255, 255, 0),       # Yellow
        "tools": (128, 128, 128),           # Gray
        "helmet": (255, 0, 0),              # Red
        "medical-suit": (0, 255, 128),      # Teal Green
        "safety-suit": (255, 140, 0),       # Dark Orange
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
        print(ppe_type, CLASS_COLORS.get(ppe_type, (255,255,255)))
        color = CLASS_COLORS.get(ppe_type, (255, 255, 255))

        x1, y1, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        x2, y2 = x1 + w, y1 + h

        # Convert to int for OpenCV
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

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


def build_alert_payload(frame_id: str, captured_at: str, camera_id: str, detections: List[Dict]) -> Dict:
    """Constructs the alert payload."""
    return {
            "frameId": frame_id,
            "capturedAt": captured_at,
            "processedAt": datetime.now(timezone.utc).isoformat(),
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

def merge_detections(detections: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
    """Merge overlapping detections of the same PPE type."""
    merged = []
    used = set()

    for i, d1 in enumerate(detections):
        if i in used:
            continue
        group = [d1]
        bbox1 = d1["bbox"]
        x1_1, y1_1 = bbox1["x"], bbox1["y"]
        x2_1 = x1_1 + bbox1["width"]
        y2_1 = y1_1 + bbox1["height"]

        for j, d2 in enumerate(detections[i + 1:], start=i + 1):
            if j in used or d2["missingPpe"] != d1["missingPpe"]:
                continue
            bbox2 = d2["bbox"]
            x1_2, y1_2 = bbox2["x"], bbox2["y"]
            x2_2 = x1_2 + bbox2["width"]
            y2_2 = y1_2 + bbox2["height"]
            if iou((x1_1, y1_1, x2_1, y2_1), (x1_2, y1_2, x2_2, y2_2)) > iou_threshold:
                group.append(d2)
                used.add(j)

        # Merge grouped detections
        if len(group) == 1:
            merged.append(group[0])
        else:
            # Merge bounding boxes and take highest confidence
            xs = [g["bbox"]["x"] for g in group]
            ys = [g["bbox"]["y"] for g in group]
            x2s = [g["bbox"]["x"] + g["bbox"]["width"] for g in group]
            y2s = [g["bbox"]["y"] + g["bbox"]["height"] for g in group]
            merged_bbox = {
                    "x": min(xs),
                    "y": min(ys),
                    "width": max(x2s) - min(xs),
                    "height": max(y2s) - min(ys)
                    }
            best_conf = max(g["confidence"] for g in group)
            merged.append({
                "missingPpe": d1["missingPpe"],
                "confidence": best_conf,
                "bbox": merged_bbox
                })
        used.add(i)

    return merged

def iou(box1, box2) -> float:
    """Compute IoU between two boxes in (x1,y1,x2,y2) format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    a1 = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    a2 = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))
    union = a1 + a2 - inter_area
    return inter_area / union if union > 0 else 0.0


def detect_missing_ppe(image_path: str) -> List[Dict]:
    """
    Run YOLO model on the image and return detections for missing PPE.
    Returns a list of dicts: {"missingPpe": str, "confidence": float, "bbox": {...}}
    Confidence is computed from region (person/head/ear) detection confidence
    reduced by the maximum overlap with the corresponding PPE boxes.
    """
    results = model.predict(image_path, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)
    out_detections: List[Dict] = []

    for r in results:
        # store detections as dicts with bbox and conf for easier use
        persons = []   # list of {"bbox": (x1,y1,x2,y2), "conf": c}
        heads = []
        ears = []
        helmets = []
        earmuffs = []

        # parse model outputs; normalize class names to lowercase
        for box in r.boxes:
            cls_id = int(box.cls)
            class_name = model.names[cls_id].lower()  # use the actual model names
            conf = float(box.conf)
            # skip any detection below the threshold (already filtered by predict conf, but safe)
            if conf < YOLO_CONFIDENCE_THRESHOLD:
                continue
            # xyxy might be a tensor-like; mapping to float should work as earlier
            x1, y1, x2, y2 = map(float, box.xyxy[0])

            entry = {"bbox": (x1, y1, x2, y2), "conf": conf}

            if class_name == "person":
                persons.append(entry)
            elif class_name == "head":
                heads.append(entry)
            elif class_name == "ear":
                ears.append(entry)
            elif class_name == "helmet":
                helmets.append(entry)
            elif class_name == "ear-mufs" or class_name == "earmuffs" or class_name == "ear_mufs":
                earmuffs.append(entry)


        # debug counts (optional)
        print(f"[DEBUG] counts -> person:{len(persons)} head:{len(heads)} ear:{len(ears)} "
              f"helmet:{len(helmets)} earmuffs:{len(earmuffs)}")

        # Helper to compute max IoU of a region vs a list of PPE boxes
        def max_overlap(region_bbox, ppe_list):
            if not ppe_list:
                return 0.0
            overlaps = [iou(region_bbox, p["bbox"]) for p in ppe_list]
            return max(overlaps) if overlaps else 0.0

        # --- Helmet missing logic ---
        # prefer checking on head regions; if no heads, check persons
        head_regions = heads if heads else persons
        # we will mark missing helmet once per region (head or person)
        for region in head_regions:
            region_bbox = region["bbox"]
            region_conf = region["conf"]
            best_iou = max_overlap(region_bbox, helmets)
            # if best_iou is below IOU_THRESHOLD we consider "no helmet covering region"
            if best_iou < IOU_THRESHOLD:
                # missing_confidence: how confident we are it's missing
                missing_conf = region_conf * (1.0 - best_iou)
                out_detections.append({
                    "missingPpe": "helmet",
                    "confidence": float(round(missing_conf, 4)),
                    "bbox": {
                        "x": float(region_bbox[0]),
                        "y": float(region_bbox[1]),
                        "width": float(region_bbox[2] - region_bbox[0]),
                        "height": float(region_bbox[3] - region_bbox[1])
                        }
                    })

        # --- Earmuff missing logic ---
        # prefer ear boxes (most precise), otherwise try head regions
        ear_regions = ears if ears else heads if heads else persons
        for region in ear_regions:
            region_bbox = region["bbox"]
            region_conf = region["conf"]
            best_iou = max_overlap(region_bbox, earmuffs)
            if best_iou < IOU_THRESHOLD:
                missing_conf = region_conf * (1.0 - best_iou)
                out_detections.append({
                    "missingPpe": "ear-mufs",
                    "confidence": float(round(missing_conf, 4)),
                    "bbox": {
                        "x": float(region_bbox[0]),
                        "y": float(region_bbox[1]),
                        "width": float(region_bbox[2] - region_bbox[0]),
                        "height": float(region_bbox[3] - region_bbox[1])
                        }
                    })


    out_detections = merge_detections(out_detections)
    return out_detections

def handle_alert(frame_id: str, captured_at: str, camera_id: str, detections: List[Dict], annotated_image: str) -> bool:
    """Handle cooldown logic and alert sending."""
    now = time.time()
    last_time = last_alert_time.get(camera_id, 0)
    if now - last_time < ALERT_COOLDOWN_SECONDS:
        logger.info(f"⏳ Skipping alert for {camera_id} (cooldown active)")
        return False

    payload = build_alert_payload(frame_id, captured_at, camera_id, detections)
    success = send_alert(payload, annotated_image)
    if success:
        last_alert_time[camera_id] = now
    return success


# ---------------------------------------------------------------------
# API Endpoint
# ---------------------------------------------------------------------
@app.post("/analyze/")
async def analyze_frame(file: UploadFile = File(...), camera_id: str = Form(...), frame_id: str = Form(...), captured_at: str = Form(...)):
    """
    Process a single camera frame, detect missing PPE, and send alert if needed.
    """
    temp_file = save_temp_file(file)
    detections = detect_missing_ppe(temp_file)
    annotated_path = None
    alert_sent = False

    if detections:
        annotated_path = draw_bounding_boxes(temp_file, detections)
        alert_sent = handle_alert(frame_id, captured_at, camera_id, detections, annotated_path)

    # Cleanup
    for path in [temp_file, annotated_path]:
        if path and os.path.exists(path):
            os.remove(path)

    return {
            "frameId": frame_id,
            "cameraId": camera_id,
            "detections": detections,
            "alertSent": alert_sent,
            }

