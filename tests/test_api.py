import io
import sys
import os
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from api.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_predict_no_file():
    response = client.post("/predict")
    assert response.status_code == 422


def test_predict_with_invalid_file():
    response = client.post("/predict", files={"file": ("test.txt", b"not an image", "text/plain")})
    assert response.status_code in (400, 422, 500, 503)


def test_gradcam_endpoint():
    img = Image.new("RGB", (224, 224), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post("/gradcam", files={"file": ("car.jpg", buf.getvalue(), "image/jpeg")})
    if response.status_code == 200:
        data = response.json()
        assert "class_name" in data
        assert "confidence" in data


def test_frontend_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
