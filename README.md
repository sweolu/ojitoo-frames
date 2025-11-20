# 🛡️ Ojitoo Frame Processor

Service responsible for analyzing camera frames, detecting missing Personal Protective Equipment (PPE) using a YOLO model, and sending alerts to the backend.

---

##  Index
1. Overview  
2. Features  
3. Project Structure  
4. Installation  
5. Configuration (.env)  
6. Running Locally  
7. Deployment (Docker & GPU)  
8. API Endpoint  
9. Alert Workflow  
10. License  

---

## Overview
The Frame Processor receives images from camera clients, runs YOLO-based detection to identify missing PPE (helmet, ear protection), annotates frames, and triggers alerts when needed.

---

## Features
- FastAPI server for frame analysis
- YOLO object detection (custom PPE model)
- Missing PPE detection & bounding boxes
- Cooldown-based alert triggering
- Sends annotated frames to backend
- GPU acceleration support (CUDA)

---

## Project Structure
```

ojitoo-frames
├── main.py               # FastAPI application
├── models/               # YOLO model files (.pt)
├── requirements.txt
├── dockerfile
├── deploy.sh
├── README.md
└── LICENSE

````

---

## Installation
```bash
git clone https://github.com/<your-org>/ojitoo-frames.git
cd ojitoo-frames
pip install -r requirements.txt
````

---

## Configuration (.env)

Create a `.env` file with:

```ini
MODEL_PATH=model/best.pt
OJITOO_BASE_URL=http://<backend-ip>:3000/api
AUTHORIZATION_TOKEN=
YOLO_CONFIDENCE_THRESHOLD=0.4
IOU_THRESHOLD=0.3
ALERT_COOLDOWN_SECONDS=30
```

---

## Running Locally

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Deployment (Docker & GPU)

```bash
sudo ./deploy.sh
```

Or manually:

```bash
docker build -t ojitoo-frames .
docker run -d -p 8000:8000 --gpus all --env-file .env ojitoo-frames
```

---

## API Endpoint

### `POST /analyze/`

| Field       | Type                    |
| ----------- | ----------------------- |
| file        | UploadFile (image/jpeg) |
| camera_id   | string                  |
| frame_id    | string                  |
| captured_at | ISO timestamp           |

**Response:**

```json
{
  "frameId": "...",
  "cameraId": "...",
  "detections": [...],
  "alertSent": true
}
```

---

## Alert Workflow

1. Frame received → YOLO inference
2. Missing PPE detected → annotated
3. Cooldown check
4. Alert payload + image sent to backend
5. Alert stored in Alert module

