import io

from fastapi.testclient import TestClient
from PIL import Image

from app import app

client = TestClient(app)


def make_jpeg():
    buffer = io.BytesIO()
    Image.new("RGB", (300, 300), "gray").save(buffer, format="JPEG")
    return buffer.getvalue()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_image():
    response = client.post("/predict", files={"file": ("part.jpg", make_jpeg(), "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] in ("defective", "normal")
    assert 0.5 <= body["confidence"] <= 1.0


def test_predict_rejects_non_image():
    response = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def test_predict_rejects_corrupt_image():
    response = client.post("/predict", files={"file": ("broken.jpg", b"not really a jpeg", "image/jpeg")})
    assert response.status_code == 400
