from fastapi import FastAPI, UploadFile, HTTPException
from ultralytics import YOLO
import cv2, numpy as np, time

app = FastAPI(title="YOLOv8n CPU Object Detection API")

# Load YOLO model (auto-downloads if not found)
model = YOLO("yolov8n.pt")

# Simple in-memory metrics
stats = {
    "total_requests": 0,
    "total_latency": 0.0,
}

@app.get("/health")
def health():
    """Health endpoint for Kubernetes readiness/liveness probes."""
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    """Return current API statistics."""
    avg_latency = (
        stats["total_latency"] / stats["total_requests"]
        if stats["total_requests"] > 0 else 0.0
    )
    return {
        "total_requests": stats["total_requests"],
        "average_latency_ms": round(avg_latency * 1000, 2),
    }

@app.post("/infer")
async def infer(file: UploadFile):
    start_time = time.time()
    try:
        image_bytes = await file.read()
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode image.")

        results = model.predict(source=img, device="cpu", verbose=False)
        detections = results[0].to_json()

        # Update metrics
        latency = time.time() - start_time
        stats["total_requests"] += 1
        stats["total_latency"] += latency

        return detections

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
