import time
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from loguru import logger

class ResNetService:
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"

        logger.info(f"[ResNet] Initializing on device={self.device}")

        self.model = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        ).to(self.device)
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=models.ResNet50_Weights.DEFAULT.transforms().mean,
                std=models.ResNet50_Weights.DEFAULT.transforms().std
            )
        ])

        self.categories = models.ResNet50_Weights.DEFAULT.meta["categories"]

    def predict(self, pil_img, workload_seconds: int = 300):
        """
        Long-running GPU workload:
        - Repeated inference loop to keep GPU busy
        - Suitable for latency / energy measurements
        """
        logger.info("[ResNet] Inference started")

        start = time.time()
        end_time = start + workload_seconds

        x = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        last_logits = None

        with torch.inference_mode():
            while time.time() < end_time:
                last_logits = self.model(x)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        probs = F.softmax(last_logits, dim=1)[0]
        topk = torch.topk(probs, k=5)

        duration = time.time() - start
        logger.info(f"[ResNet] Inference finished after {duration:.2f}s")

        return {
            "model": "resnet50",
            "device": self.device,
            "runtime_seconds": round(duration, 2),
            "top5": [
                {"label": self.categories[i], "prob": float(p)}
                for p, i in zip(topk.values.tolist(), topk.indices.tolist())
            ],
        }
