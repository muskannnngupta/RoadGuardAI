# RoadGuard AI — YOLOv8 Inference Microservice

Dedicated FastAPI microservice for high-throughput road distress detection using fine-tuned Ultralytics YOLOv8 weights (`D00`, `D10`, `D20`, `D40`).

---

## 1. Overview & Architecture

- **Single Model Load:** The YOLOv8 model weights are loaded into memory once at service startup via FastAPI's lifespan event handler.
- **Two-Tier Confidence Filtering:**
  - `CONF_UPPER` (Default `0.50`): Detections at or above this threshold are flagged as `"suggested"` (pre-selected for inspector verification).
  - `CONF_LOWER` (Default `0.25`): Detections between `CONF_LOWER` and `CONF_UPPER` are flagged as `"low_confidence"`. Detections strictly below `CONF_LOWER` are discarded.
  - If no detections meet `CONF_LOWER`, the service returns HTTP 200 with `status: "no_damage_detected"` (never an error).
- **Validation Guardrails:** Rejects unsupported file types, empty files, corrupt/truncated bytes, and oversized uploads with clear `4xx` error codes.

> [!IMPORTANT]
> **Threshold Calibration Notice:**
> The default thresholds (`CONF_LOWER = 0.25` and `CONF_UPPER = 0.50`) are initial placeholders. They must be calibrated against the validation set precision-recall / F1 curves generated during model training (`ml/kaggle_train.py`) to align with municipal dispatch risk tolerances.

---

## 2. API Endpoints

### `GET /health`
Returns service operational status, model loaded status, and active configuration.

**Response Example (200 OK):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "C:\\Users\\...\\inference-service\\models\\best.pt",
  "model_version": "yolov8n-rdd2022-v1",
  "thresholds": {
    "conf_lower": 0.25,
    "conf_upper": 0.50
  }
}
```

---

### `POST /detect`
Accepts multipart form-data image upload (`.jpg`, `.jpeg`, `.png`) and executes YOLO inference.

**Request:**
- `Content-Type: multipart/form-data`
- Body parameter: `file` (binary image data)

**Response Example (Damage Detected - 200 OK):**
```json
{
  "status": "damage_detected",
  "model_version": "yolov8n-rdd2022-v1",
  "thresholds": {
    "conf_lower": 0.25,
    "conf_upper": 0.50
  },
  "image_size": {
    "width": 1920,
    "height": 1080
  },
  "detections_count": 2,
  "detections": [
    {
      "class_name": "D40",
      "confidence": 0.8421,
      "bbox": [450.25, 620.10, 890.75, 910.40],
      "flag": "suggested"
    },
    {
      "class_name": "D00",
      "confidence": 0.3845,
      "bbox": [120.50, 480.00, 310.20, 560.80],
      "flag": "low_confidence"
    }
  ]
}
```

**Response Example (Clean Road / No Damage - 200 OK):**
```json
{
  "status": "no_damage_detected",
  "model_version": "yolov8n-rdd2022-v1",
  "thresholds": {
    "conf_lower": 0.25,
    "conf_upper": 0.50
  },
  "image_size": {
    "width": 1280,
    "height": 720
  },
  "detections_count": 0,
  "detections": []
}
```

---

## 3. Configuration (Environment Variables)

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `MODEL_PATH` | `models/best.pt` | Path to fine-tuned YOLOv8 weights file |
| `CONF_LOWER` | `0.25` | Lower threshold below which detections are discarded |
| `CONF_UPPER` | `0.50` | Upper threshold for automated `"suggested"` flag |
| `MODEL_VERSION` | `yolov8n-rdd2022-v1` | Identifier version string for audit trails |
| `MAX_FILE_SIZE_MB` | `15` | Maximum permitted upload size in MB |
| `HOST` | `0.0.0.0` | Binding host address |
| `PORT` | `8000` | Binding network port |

---

## 4. Local Execution Steps

### 1. Create and Activate Virtual Environment
```bash
cd inference-service
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Linux / macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Ensure Weights File is Present
Ensure your trained model weights exist at `models/best.pt`:
```text
inference-service/
├── models/
│   └── best.pt
```

### 4. Start the Uvicorn Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger UI documentation is available at:
👉 **`http://localhost:8000/docs`**

---

## 5. Docker Deployment

### Build the Image
```bash
docker build -t roadguard-inference:latest -f Dockerfile .
```

### Run the Container
```bash
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -e CONF_LOWER=0.25 \
  -e CONF_UPPER=0.50 \
  --name roadguard-inference \
  roadguard-inference:latest
```

---

## 6. Running Automated Tests

Run the test suite using `pytest`:
```bash
pytest tests/ -v
```

The test suite validates:
- `GET /health` operational health and configuration schema.
- `POST /detect` valid image processing with confidence tier filtering.
- `POST /detect` clean `"no_damage_detected"` response when no damage is found.
- `400 Bad Request` handling for invalid extensions (`.txt`, `.pdf`).
- `400 Bad Request` handling for corrupted/unreadable image streams.
- `400 Bad Request` handling for empty (0-byte) files.
- `413 Request Entity Too Large` handling for oversized uploads.
- `503 Service Unavailable` error handling if model weights are missing or failed to initialize.
