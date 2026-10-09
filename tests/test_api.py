import io
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app as A  # noqa: E402


class FakeDetector:
    def predict(self, image, conf):
        objects = [{"class": "pill", "confidence": 0.9, "box": [1, 1, 10, 10]},
                   {"class": "pill", "confidence": 0.5, "box": [20, 20, 30, 30]},
                   {"class": "blister", "confidence": 0.7, "box": [0, 0, 50, 50]}]
        return [o for o in objects if o["confidence"] >= conf]


A.app.dependency_overrides[A.get_detector] = lambda: FakeDetector()
client = TestClient(A.app)


def png_bytes(w=64, h=48):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 200, 200)).save(buf, "PNG")
    return buf.getvalue()


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_returns_objects_and_size():
    r = client.post("/predict", files={"file": ("a.png", png_bytes(), "image/png")})
    body = r.json()
    assert r.status_code == 200 and body["count"] == 3 and body["width"] == 64


def test_count_respects_confidence():
    r = client.post("/count?conf=0.6", files={"file": ("a.png", png_bytes(), "image/png")})
    assert r.json() == {"total": 2, "by_class": {"pill": 1, "blister": 1}}


def test_bad_image_is_400():
    r = client.post("/predict", files={"file": ("a.png", b"not an image", "image/png")})
    assert r.status_code == 400


def test_exif_rotation_is_applied():
    buf = io.BytesIO()
    img = Image.new("RGB", (64, 48))
    exif = img.getexif()
    exif[274] = 6  # «повернуть на 90°»
    img.save(buf, "JPEG", exif=exif.tobytes())
    body = client.post("/predict", files={"file": ("a.jpg", buf.getvalue(), "image/jpeg")}).json()
    assert (body["width"], body["height"]) == (48, 64)
