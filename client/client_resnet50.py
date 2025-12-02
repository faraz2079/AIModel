import argparse
import time
import json
import requests
import numpy as np
from PIL import Image

# ImageNet mean/std in RGB, as used by TF/ResNet
IMAGENET_MEAN = np.array([123.68, 116.779, 103.939])
IMAGENET_STD = np.array([58.393, 57.12, 57.375])

def preprocess(img_path):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    arr = np.array(img).astype(np.float32)
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.expand_dims(arr, 0)
    return arr

def load_labels():
    with open("imagenet_labels.json", "r") as f:
        return json.load(f)

def infer(url, arr):
    payload = {
        "inputs": [{
            "name": "input_1",
            "shape": arr.shape,
            "datatype": "FP32",
            "data": arr.flatten().tolist()
        }],
        "outputs": [{"name": "predictions"}]
    }

    t0 = time.time()
    r = requests.post(f"{url}/v2/models/resnet50/infer",
                      headers={"Content-Type": "application/json"},
                      data=json.dumps(payload))
    latency = (time.time() - t0) * 1000
    return r.json(), latency

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--topk", default=5, type=int)
    args = parser.parse_args()

    arr = preprocess(args.image)
    labels = load_labels()

    resp, latency = infer(args.url, arr)
    preds = np.array(resp["outputs"][0]["data"])

    top_indices = preds.argsort()[::-1][:args.topk]
    print(f"\nInference latency: {latency:.2f} ms\n")
    print("Top predictions:")

    for i in top_indices:
        print(f"{labels[i]:20s}   score={preds[i]:.4f}")

if __name__ == "__main__":
    main()
