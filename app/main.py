from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal

from scheduler import GPUScheduler
from models.resnet_service import ResNetService
from models.yolo_service import YOLOService
from utils.image_io import b64_to_pil_image

app = FastAPI(title="Hybrid GPU Inference")

scheduler = GPUScheduler()
resnet = ResNetService()
yolo = YOLOService()

class InferenceRequest(BaseModel):
    image_b64: str
    task: Literal["classification", "detection"]

@app.post("/infer")
async def infer(req: InferenceRequest):
    img = b64_to_pil_image(req.image_b64)

    if req.task == "classification":
        return await scheduler.run(lambda: resnet.predict(img))

    if req.task == "detection":
        return await scheduler.run(lambda: yolo.predict(img))

    return {"error": "Unknown task"}
