import logging
from pathlib import Path
from typing import List, Optional
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from app.config import settings
from app.schemas import Detection

logger = logging.getLogger("roadguard.inference")


class DamageDetector:
    """
    Singleton wrapper for Ultralytics YOLOv8 road damage detection model.
    Loads weights once at application startup.
    """

    def __init__(self, model_path: Optional[str] = None, model_version: Optional[str] = None):
        self.raw_model_path = model_path or settings.MODEL_PATH
        self.model_version = model_version or settings.MODEL_VERSION
        self.resolved_model_path: Optional[str] = None
        self.model: Optional[YOLO] = None
        self._load_model()

    def _resolve_path(self, target_path: str) -> Optional[Path]:
        """Resolves relative or absolute model path across project and docker environments."""
        p = Path(target_path)
        if p.is_absolute() and p.exists():
            return p

        candidates = [
            p,
            Path.cwd() / p,
            Path(__file__).resolve().parent.parent / p,
            Path(__file__).resolve().parent.parent.parent / p,
            Path(__file__).resolve().parent.parent / "models" / p.name,
        ]

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()

        return None

    def _load_model(self) -> None:
        """Loads YOLO model weights if path exists."""
        if YOLO is None:
            logger.warning(
                "Ultralytics package is not installed in the environment. "
                "Inference will be disabled until ultralytics is available."
            )
            self.model = None
            self.resolved_model_path = self.raw_model_path
            return

        resolved = self._resolve_path(self.raw_model_path)
        if resolved:
            try:
                logger.info(f"Loading YOLOv8 weights from: {resolved}")
                self.model = YOLO(str(resolved))
                self.resolved_model_path = str(resolved)
                logger.info(f"Model successfully loaded. Classes registered: {self.model.names}")
            except Exception as exc:
                logger.error(f"Error loading model from {resolved}: {exc}", exc_info=True)
                self.model = None
                self.resolved_model_path = str(resolved)
        else:
            logger.warning(
                f"Model weights not found at '{self.raw_model_path}'. "
                f"Inference requests will fail until valid weights are provided."
            )
            self.model = None
            self.resolved_model_path = self.raw_model_path

    @property
    def is_loaded(self) -> bool:
        """Returns True if the underlying YOLO model is loaded and ready."""
        return self.model is not None

    def predict(
        self,
        image: Image.Image,
        conf_lower: float = settings.CONF_LOWER,
        conf_upper: float = settings.CONF_UPPER
    ) -> List[Detection]:
        """
        Executes YOLO inference on an input PIL image.
        Filters detections by conf_lower and tags with 'suggested' or 'low_confidence'.
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError(
                f"Model is not loaded (path: {self.resolved_model_path or self.raw_model_path}). "
                f"Cannot execute inference."
            )

        # Execute YOLO inference using conf_lower as initial pre-filter
        results = self.model.predict(
            source=image,
            conf=conf_lower,
            verbose=False
        )

        detections: List[Detection] = []
        if not results or len(results) == 0:
            return detections

        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        names_map = self.model.names or {}

        for box in boxes:
            conf = float(box.conf[0].item())
            # Explicit threshold filter: discard if below conf_lower
            if conf < conf_lower:
                continue

            cls_id = int(box.cls[0].item())
            class_name = names_map.get(cls_id, f"D{cls_id}")

            # Extract bounding box in pixel coordinates [x1, y1, x2, y2]
            xyxy = [round(float(coord), 2) for coord in box.xyxy[0].tolist()]

            # Determine confidence tier flag
            flag = "suggested" if conf >= conf_upper else "low_confidence"

            detections.append(
                Detection(
                    class_name=str(class_name),
                    confidence=round(conf, 4),
                    bbox=xyxy,
                    flag=flag
                )
            )

        return detections


# Global singleton instance
_detector_instance: Optional[DamageDetector] = None


def get_detector() -> DamageDetector:
    """Provides or initializes the global DamageDetector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = DamageDetector()
    return _detector_instance


def set_detector(detector: Optional[DamageDetector]) -> None:
    """Overrides global detector instance (primarily for unit tests/mocking)."""
    global _detector_instance
    _detector_instance = detector
