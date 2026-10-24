import io
from typing import Generator
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.model import DamageDetector, set_detector
from app.schemas import Detection


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Test client fixture for FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    """Creates a valid in-memory JPEG test image (640x640)."""
    img = Image.new("RGB", (640, 640), color=(128, 128, 128))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def sample_png_bytes() -> bytes:
    """Creates a valid in-memory PNG test image (320x240)."""
    img = Image.new("RGB", (320, 240), color=(200, 100, 50))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


class MockDamageDetector:
    """Mock detector implementing DamageDetector interface for unit tests."""

    def __init__(self, is_loaded: bool = True, mock_detections: list = None):
        self._is_loaded = is_loaded
        self.raw_model_path = "models/best.pt"
        self.resolved_model_path = "/app/models/best.pt"
        self.model_version = "yolov8n-rdd2022-v1"
        self.mock_detections = mock_detections if mock_detections is not None else []

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def predict(self, image: Image.Image, conf_lower: float = 0.25, conf_upper: float = 0.50):
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")
        # Filter detections based on conf_lower and assign flags dynamically
        filtered = []
        for d in self.mock_detections:
            if d.confidence >= conf_lower:
                flag = "suggested" if d.confidence >= conf_upper else "low_confidence"
                filtered.append(
                    Detection(
                        class_name=d.class_name,
                        confidence=d.confidence,
                        bbox=d.bbox,
                        flag=flag
                    )
                )
        return filtered


@pytest.fixture
def mock_detector_with_detections():
    """Sets a mock detector returning predefined detections."""
    detections = [
        Detection(class_name="D40", confidence=0.82, bbox=[100.0, 150.0, 250.0, 300.0], flag="suggested"),
        Detection(class_name="D00", confidence=0.38, bbox=[400.0, 420.0, 480.0, 550.0], flag="low_confidence"),
        Detection(class_name="D20", confidence=0.15, bbox=[10.0, 20.0, 80.0, 90.0], flag="low_confidence"),
    ]
    mock = MockDamageDetector(is_loaded=True, mock_detections=detections)
    set_detector(mock)
    yield mock
    set_detector(None)


@pytest.fixture
def mock_detector_empty():
    """Sets a mock detector returning zero detections."""
    mock = MockDamageDetector(is_loaded=True, mock_detections=[])
    set_detector(mock)
    yield mock
    set_detector(None)


@pytest.fixture
def mock_detector_unloaded():
    """Sets an unloaded mock detector to test error handling."""
    mock = MockDamageDetector(is_loaded=False)
    set_detector(mock)
    yield mock
    set_detector(None)
