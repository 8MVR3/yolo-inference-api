"""HTTP-сервис для детектора YOLO: принимает фото, отдаёт найденные объекты и их количество."""
from __future__ import annotations

import io
import os
from collections import Counter
from functools import lru_cache

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
IMGSZ = int(os.getenv("IMGSZ", "640"))


class YoloDetector:
    def __init__(self, weights: str):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.names = self.model.names

    def predict(self, image: Image.Image, conf: float) -> list[dict]:
        result = self.model.predict(image, conf=conf, imgsz=IMGSZ, verbose=False)[0]
        objects = []
        for box in result.boxes:
            x1, y1, x2, y2 = (round(v, 1) for v in box.xyxy[0].tolist())
            objects.append({"class": self.names[int(box.cls)], "confidence": round(float(box.conf), 3),
                            "box": [x1, y1, x2, y2]})
        return objects


@lru_cache(maxsize=1)
def get_detector():
    return YoloDetector(os.getenv("MODEL_PATH", "weights/best.pt"))


app = FastAPI(title="YOLO inference API", version="0.1.0")


async def read_image(file: UploadFile) -> Image.Image:
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"Файл больше {MAX_BYTES // 1024 // 1024} МБ")
    try:
        image = Image.open(io.BytesIO(data))
        # Телефоны пишут поворот в EXIF: без этого модель увидит фото боком.
        return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Не удалось прочитать изображение")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = Query(0.4, ge=0.01, le=1.0),
                  detector=Depends(get_detector)):
    image = await read_image(file)
    objects = detector.predict(image, conf)
    return {"width": image.width, "height": image.height, "count": len(objects), "objects": objects}


@app.post("/count")
async def count(file: UploadFile = File(...), conf: float = Query(0.4, ge=0.01, le=1.0),
                detector=Depends(get_detector)):
    image = await read_image(file)
    objects = detector.predict(image, conf)
    return {"total": len(objects), "by_class": dict(Counter(o["class"] for o in objects))}
