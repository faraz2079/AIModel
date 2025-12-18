import time
import torch
from ultralytics import YOLO

class YOLOService:
    def __init__(self, model_name: str = "yolov8n.pt", device: str = "cuda"):
        self.device = device
        self.model = YOLO(model_name)
        # Ultralytics handles device internally, but we keep device info for logging.

    def predict(self, pil_img):
        t0 = time.time()
        results = self.model.predict(pil_img, verbose=False)
        # Force GPU sync for accurate timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt_ms = (time.time() - t0) * 1000.0

        r0 = results[0]
        boxes = []
        if r0.boxes is not None:
            for b in r0.boxes:
                xyxy = b.xyxy[0].tolist()
                conf = float(b.conf[0].item()) if b.conf is not None else None
                cls = int(b.cls[0].item()) if b.cls is not None else None
                boxes.append({"xyxy": xyxy, "conf": conf, "cls": cls})

        return {
            "model": "yolov8n",
            "num_detections": len(boxes),
            "detections": boxes,
            "inference_time_ms": round(dt_ms, 2),
        }
