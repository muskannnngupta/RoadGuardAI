from typing import List, Literal
from pydantic import BaseModel, Field


class Detection(BaseModel):
    """Individual road distress bounding box detection."""
    class_name: str = Field(
        ...,
        description="Distress category (e.g., D00, D10, D20, D40)"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence score between 0.0 and 1.0"
    )
    bbox: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Pixel coordinates in format [x1, y1, x2, y2] (top-left, bottom-right)"
    )
    flag: Literal["suggested", "low_confidence"] = Field(
        ...,
        description="'suggested' if confidence >= CONF_UPPER, 'low_confidence' if CONF_LOWER <= confidence < CONF_UPPER"
    )


class ThresholdsConfig(BaseModel):
    """Active confidence threshold settings."""
    conf_lower: float = Field(..., description="Minimum threshold required to retain detection")
    conf_upper: float = Field(..., description="Threshold at or above which detection is marked suggested")


class ImageMetadata(BaseModel):
    """Input image dimensional metadata."""
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")


class DetectionResponse(BaseModel):
    """API response schema for /detect endpoint."""
    status: Literal["damage_detected", "no_damage_detected"] = Field(
        ...,
        description="'damage_detected' if >=1 valid box found, otherwise 'no_damage_detected'"
    )
    model_version: str = Field(..., description="Model version tag")
    thresholds: ThresholdsConfig = Field(..., description="Threshold configuration used for filtering")
    image_size: ImageMetadata = Field(..., description="Processed image dimensions")
    detections_count: int = Field(..., description="Total count of retained detections")
    detections: List[Detection] = Field(default_factory=list, description="List of filtered distress detections")


class HealthResponse(BaseModel):
    """API response schema for /health endpoint."""
    status: str = Field(..., description="Service health status string")
    model_loaded: bool = Field(..., description="True if weights are loaded and ready for inference")
    model_path: str = Field(..., description="Configured or resolved model filesystem path")
    model_version: str = Field(..., description="Model version tag")
    thresholds: ThresholdsConfig = Field(..., description="Active confidence thresholds")
