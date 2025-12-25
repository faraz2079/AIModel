from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
from loguru import logger

from scheduler import GPUScheduler
from models.resnet_service import ResNetService
from models.yolo_service import YOLOService
from utils.image_io import b64_to_pil_image


# --------------------
# App initialization
# --------------------
app = FastAPI(title="Hybrid GPU Inference (Long Workload)")

scheduler = GPUScheduler()
resnet = ResNetService()
yolo = YOLOService()


# --------------------
# Request schema
# --------------------
class InferenceRequest(BaseModel):
    task: Literal["classification", "detection"]
    image_b64: str
    workload_seconds: int = 300  # default: 5 minutes


# --------------------
# API endpoint
# --------------------
@app.post("/infer")
async def infer(req: InferenceRequest):
    logger.info(
        f"[API] Request received "
        f"task={req.task}, workload={req.workload_seconds}s"
    )

    img = b64_to_pil_image(req.image_b64)

    if req.task == "classification":

        def run_resnet():
            logger.info("[RESNET] Starting GPU workload")
            return resnet.predict(img, req.workload_seconds)

        return await scheduler.run(run_resnet)

    if req.task == "detection":

        def run_yolo():
            logger.info("[YOLO] Starting GPU workload")
            return yolo.predict(img, req.workload_seconds)

        return await scheduler.run(run_yolo)

    return {"error": "Invalid task"}
