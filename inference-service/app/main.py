import io
import logging
from contextlib import asynccontextmanager
from typing import Set
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.config import settings
from app.model import get_detector
from app.schemas import (
    DetectionResponse,
    HealthResponse,
    ImageMetadata,
    ThresholdsConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("roadguard.api")

ALLOWED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png"}
ALLOWED_CONTENT_TYPES: Set[str] = {
    "image/jpeg",
    "image/png",
    "image/pjpeg",
    "image/x-png"
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: initializes YOLO model weights on service startup."""
    logger.info("Initializing RoadGuard AI Inference Service...")
    detector = get_detector()
    if detector.is_loaded:
        logger.info(f"Model successfully warmed up: {detector.resolved_model_path}")
    else:
        logger.warning("Service started, but model weights were not found or failed to load.")
    yield
    logger.info("Shutting down RoadGuard AI Inference Service...")


app = FastAPI(
    title="RoadGuard AI - Inference Service",
    description="High-throughput YOLOv8 road damage detection microservice for D00, D10, D20, and D40 distress classes.",
    version=settings.MODEL_VERSION,
    lifespan=lifespan
)

# Enable CORS for frontend and internal microservice communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service and Model Health Status",
    tags=["Monitoring"]
)
async def health_check():
    """
    Checks operational health of the inference microservice and status of the YOLO model.
    """
    detector = get_detector()
    is_ready = detector.is_loaded

    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        model_loaded=is_ready,
        model_path=detector.resolved_model_path or detector.raw_model_path,
        model_version=settings.MODEL_VERSION,
        thresholds=ThresholdsConfig(
            conf_lower=settings.CONF_LOWER,
            conf_upper=settings.CONF_UPPER
        )
    )


@app.post(
    "/detect",
    response_model=DetectionResponse,
    summary="Detect Road Distress from Image Upload",
    tags=["Inference"]
)
async def detect_road_damage(file: UploadFile = File(...)):
    """
    Accepts an image file (.jpg or .png), performs YOLOv8 damage detection,
    and returns detected distress bounding boxes with confidence scores and confidence tier flags.

    - Discards detections with confidence < CONF_LOWER.
    - Tags detections with confidence >= CONF_UPPER as 'suggested'.
    - Tags detections between CONF_LOWER and CONF_UPPER as 'low_confidence'.
    - Returns HTTP 200 with status 'no_damage_detected' if no bounding boxes satisfy CONF_LOWER.
    """
    # 1. Validate File Extension
    filename = (file.filename or "").lower()
    has_valid_ext = any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS)
    has_valid_mime = file.content_type in ALLOWED_CONTENT_TYPES if file.content_type else False

    if not has_valid_ext and not has_valid_mime:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file format '{file.filename}'. "
                f"Only JPEG (.jpg, .jpeg) and PNG (.png) images are permitted."
            )
        )

    # 2. Validate File Size
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty (0 bytes)."
        )

    if len(contents) > max_bytes:
        http_413 = getattr(status, "HTTP_413_CONTENT_TOO_LARGE", status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
        raise HTTPException(
            status_code=http_413,
            detail=f"Uploaded image exceeds the maximum permitted size of {settings.MAX_FILE_SIZE_MB}MB."
        )

    # 3. Validate Image Integrity (Prevent corrupt/truncated files)
    try:
        # Step 3a: Verify byte stream integrity
        probe_image = Image.open(io.BytesIO(contents))
        probe_image.verify()

        # Step 3b: Load image and convert to RGB (standard format for YOLO)
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        width, height = image.size
    except Exception as exc:
        logger.warning(f"Failed to decode uploaded image '{file.filename}': {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Corrupted or unreadable image file. Unable to decode image content."
        )

    # 4. Check Model Readiness
    detector = get_detector()
    if not detector.is_loaded:
        logger.error("Inference attempted but model is not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference model is currently unavailable or failed to load. Please check /health."
        )

    # 5. Execute Model Inference
    try:
        detections = detector.predict(
            image=image,
            conf_lower=settings.CONF_LOWER,
            conf_upper=settings.CONF_UPPER
        )
    except Exception as exc:
        logger.error(f"Inference execution failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal inference error: {str(exc)}"
        )

    # 6. Build and Return Response
    has_detections = len(detections) > 0
    response_status = "damage_detected" if has_detections else "no_damage_detected"

    return DetectionResponse(
        status=response_status,
        model_version=settings.MODEL_VERSION,
        thresholds=ThresholdsConfig(
            conf_lower=settings.CONF_LOWER,
            conf_upper=settings.CONF_UPPER
        ),
        image_size=ImageMetadata(width=width, height=height),
        detections_count=len(detections),
        detections=detections
    )
