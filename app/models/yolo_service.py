import time
import torch
from ultralytics import YOLO
from loguru import logger

class YOLOService:
    def __init__(self, model_name: str = "yolov8n.pt"):
        logger.info(f"[YOLO] Loading model {model_name}")
        self.model = YOLO(model_name)

    def predict(self, pil_img, workload_seconds: int = 300):
        """
        Long-running GPU workload using YOLO inference loop
        """
        logger.info("[YOLO] Inference started")

        start = time.time()
        end_time = start + workload_seconds

        last_results = None

        while time.time() < end_time:
            last_results = self.model.predict(
                pil_img,
                verbose=False,
                device=0 if torch.cuda.is_available() else "cpu"
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        duration = time.time() - start
        r0 = last_results[0]

        boxes = []
        if r0.boxes is not None:
            for b in r0.boxes:
                boxes.append({
                    "xyxy": b.xyxy[0].tolist(),
                    "conf": float(b.conf[0]),
                    "cls": int(b.cls[0]),
                })

        logger.info(f"[YOLO] Inference finished after {duration:.2f}s")

        return {
            "model": "yolov8n",
            "runtime_seconds": round(duration, 2),
            "num_detections": len(boxes),
            "detections": boxes,
        }
