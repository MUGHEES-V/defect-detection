import io
import logging
import os
import time

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image

MODEL_PATH = os.getenv("MODEL_PATH", "model.onnx")
CLASS_NAMES = ["defective", "normal"]
IMG_SIZE = 224
MAX_BYTES = 5 * 1024 * 1024
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("defect-api")

app = FastAPI(title="Defect Detection API")
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
log.info("model loaded from %s", MODEL_PATH)


def preprocess(image):
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = np.asarray(image, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(file: UploadFile):
    start = time.perf_counter()

    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "File must be an image (JPEG or PNG).")
    data = file.file.read(MAX_BYTES + 1)
    if len(data) == 0:
        raise HTTPException(400, "File is empty.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image is larger than 5 MB.")
    try:
        x = preprocess(Image.open(io.BytesIO(data)))
    except Exception:
        log.warning("could not decode file=%s", file.filename)
        raise HTTPException(400, "Could not read the file as an image.")

    try:
        logits = session.run(None, {"input": x})[0][0]
    except Exception:
        log.exception("inference failed for file=%s", file.filename)
        raise HTTPException(500, "Inference failed.")

    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    idx = int(probs.argmax())
    result = {"predicted_class": CLASS_NAMES[idx], "confidence": round(float(probs[idx]), 4)}

    latency_ms = (time.perf_counter() - start) * 1000
    log.info("file=%s class=%s confidence=%.4f latency_ms=%.1f",
             file.filename, result["predicted_class"], result["confidence"], latency_ms)
    return result
