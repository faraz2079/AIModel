from PIL import Image
import numpy as np
import requests
import json

# Load ImageNet labels
with open("imagenet_labels.json") as f:
    labels = json.load(f)

# Load and preprocess image
img = Image.open("scripts/kitten_small.jpg").convert("RGB")
img = img.resize((224, 224))

arr = np.array(img).astype(np.float32) / 255.0
arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
arr = np.expand_dims(arr, 0)

# Prepare Triton payload
payload = {
    "inputs": [
        {
            "name": "input_1",
            "shape": arr.shape,
            "datatype": "FP32",
            "data": arr.flatten().tolist()
        }
    ],
    "outputs": [{"name": "predictions"}]
}

# Send request
response = requests.post(
    "http://localhost:8000/v2/models/resnet50/infer",
    headers={"Content-Type": "application/json"},
    data=json.dumps(payload)
)

preds = np.array(response.json()["outputs"][0]["data"])

# Top-5 predictions
top5_idx = preds.argsort()[-5:][::-1]

print("Top-5 predictions:")
for i in top5_idx:
    print(f"{labels[i]}  (score = {preds[i]:.4f})")
