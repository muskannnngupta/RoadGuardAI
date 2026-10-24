import os

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    from pydantic import BaseModel as BaseSettings
    SettingsConfigDict = None

from pydantic import Field, field_validator, ConfigDict


class Settings(BaseSettings):
    """
    Configuration settings for RoadGuard AI YOLO Inference Service.
    Configurable via environment variables or .env file.
    """
    if SettingsConfigDict:
        model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    else:
        model_config = ConfigDict(extra="ignore")

    MODEL_PATH: str = Field(
        default=os.getenv("MODEL_PATH", "models/best.pt"),
        description="Filesystem path to trained YOLOv8 model weights (.pt)"
    )
    # Default thresholds are initial placeholders; calibrate against validation PR/F1 curve
    CONF_LOWER: float = Field(
        default=float(os.getenv("CONF_LOWER", "0.25")),
        description="Lower confidence threshold; detections below this are discarded"
    )
    CONF_UPPER: float = Field(
        default=float(os.getenv("CONF_UPPER", "0.50")),
        description="Upper confidence threshold; detections at/above this are flagged as suggested"
    )
    MODEL_VERSION: str = Field(
        default=os.getenv("MODEL_VERSION", "yolov8n-rdd2022-v1"),
        description="Model architecture and version identifier"
    )
    MAX_FILE_SIZE_MB: int = Field(
        default=int(os.getenv("MAX_FILE_SIZE_MB", "15")),
        description="Maximum allowed image upload size in megabytes"
    )
    HOST: str = Field(
        default=os.getenv("HOST", "0.0.0.0"),
        description="FastAPI service binding host"
    )
    PORT: int = Field(
        default=int(os.getenv("PORT", "8000")),
        description="FastAPI service binding port"
    )

    @field_validator("CONF_LOWER", "CONF_UPPER")
    @classmethod
    def validate_confidence_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Confidence threshold must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("CONF_UPPER")
    @classmethod
    def validate_threshold_order(cls, v: float, info) -> float:
        lower = info.data.get("CONF_LOWER", 0.0)
        if v < lower:
            raise ValueError(f"CONF_UPPER ({v}) must be greater than or equal to CONF_LOWER ({lower})")
        return v




settings = Settings()
