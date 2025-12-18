import time
import torch
import torch.nn.functional as F
from torchvision import models, transforms

class ResNetService:
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(self.device)
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

    @torch.inference_mode()
    def predict(self, pil_img):
        t0 = time.time()
        x = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        logits = self.model(x)
        probs = F.softmax(logits, dim=1)[0]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt_ms = (time.time() - t0) * 1000.0

        topk = torch.topk(probs, k=5)
        results = []
        for p, idx in zip(topk.values.tolist(), topk.indices.tolist()):
            results.append({
                "label": self.categories[idx],
                "prob": float(p),
            })

        return {
            "model": "resnet50",
            "top5": results,
            "inference_time_ms": round(dt_ms, 2),
            "device": self.device,
        }
