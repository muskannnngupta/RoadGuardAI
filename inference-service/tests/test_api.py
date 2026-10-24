import pytest
from fastapi.testclient import TestClient

from app.config import settings


def test_health_endpoint(client: TestClient, mock_detector_empty):
    """Test GET /health returns 200 with required keys and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["model_loaded"] is True
    assert "model_path" in data
    assert "model_version" in data
    assert "thresholds" in data
    assert data["thresholds"]["conf_lower"] == settings.CONF_LOWER
    assert data["thresholds"]["conf_upper"] == settings.CONF_UPPER


def test_detect_valid_image_with_detections(client: TestClient, sample_jpeg_bytes: bytes, mock_detector_with_detections):
    """Test POST /detect with a valid JPEG image and returned detections."""
    response = client.post(
        "/detect",
        files={"file": ("road_surface.jpg", sample_jpeg_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "damage_detected"
    assert data["model_version"] == settings.MODEL_VERSION
    assert data["image_size"]["width"] == 640
    assert data["image_size"]["height"] == 640

    # Detections check: D40 (0.82) -> suggested, D00 (0.38) -> low_confidence, D20 (0.15) -> discarded (<0.25)
    detections = data["detections"]
    assert len(detections) == 2
    assert data["detections_count"] == 2

    # Verify first detection (D40 - Pothole)
    d1 = detections[0]
    assert d1["class_name"] == "D40"
    assert d1["confidence"] == 0.82
    assert d1["flag"] == "suggested"
    assert len(d1["bbox"]) == 4

    # Verify second detection (D00 - Crack)
    d2 = detections[1]
    assert d2["class_name"] == "D00"
    assert d2["confidence"] == 0.38
    assert d2["flag"] == "low_confidence"


def test_detect_no_detection_case(client: TestClient, sample_png_bytes: bytes, mock_detector_empty):
    """
    Test POST /detect when no damages are detected.
    Must return HTTP 200 with status 'no_damage_detected' (not an error).
    """
    response = client.post(
        "/detect",
        files={"file": ("clean_road.png", sample_png_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "no_damage_detected"
    assert data["detections"] == []
    assert data["detections_count"] == 0
    assert data["image_size"]["width"] == 320
    assert data["image_size"]["height"] == 240


def test_detect_invalid_file_extension(client: TestClient):
    """Test POST /detect with unsupported file types returns HTTP 400."""
    fake_txt_bytes = b"Hello world! This is a plain text file."
    response = client.post(
        "/detect",
        files={"file": ("report.txt", fake_txt_bytes, "text/plain")}
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_detect_corrupt_image(client: TestClient):
    """Test POST /detect with unreadable/corrupted image bytes returns HTTP 400."""
    corrupted_bytes = b"CORRUPTED_BINARY_DATA_THAT_IS_NOT_A_VALID_IMAGE"
    response = client.post(
        "/detect",
        files={"file": ("broken.jpg", corrupted_bytes, "image/jpeg")}
    )
    assert response.status_code == 400
    assert "Corrupted or unreadable image file" in response.json()["detail"]


def test_detect_empty_file(client: TestClient):
    """Test POST /detect with 0-byte file returns HTTP 400."""
    response = client.post(
        "/detect",
        files={"file": ("empty.jpg", b"", "image/jpeg")}
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_detect_file_too_large(client: TestClient, monkeypatch):
    """Test POST /detect with file exceeding MAX_FILE_SIZE_MB returns HTTP 413."""
    # Temporarily set max size to 1 MB for testing
    monkeypatch.setattr(settings, "MAX_FILE_SIZE_MB", 1)

    oversized_bytes = b"x" * (1 * 1024 * 1024 + 100)  # > 1MB
    response = client.post(
        "/detect",
        files={"file": ("giant_image.jpg", oversized_bytes, "image/jpeg")}
    )
    assert response.status_code == 413
    assert "exceeds the maximum permitted size" in response.json()["detail"]


def test_detect_model_unavailable(client: TestClient, sample_jpeg_bytes: bytes, mock_detector_unloaded):
    """Test POST /detect when model failed to load returns HTTP 503."""
    response = client.post(
        "/detect",
        files={"file": ("road.jpg", sample_jpeg_bytes, "image/jpeg")}
    )
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()
