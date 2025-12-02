import argparse, time, json, requests, numpy as np
from PIL import Image
from datetime import datetime

IMAGENET_MEAN = np.array([123.68, 116.779, 103.939])
IMAGENET_STD = np.array([58.393, 57.12, 57.375])

def preprocess(img_path):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    arr = np.array(img).astype(np.float32)
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(arr, 0)

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
    return latency

def mixed_load_pattern(t):
    """
    Returns QPS (queries per second) depending on time segment.
    t is elapsed time in seconds.
    0-60s: steady (2 QPS)
    60-120s: burst (6 QPS)
    120-180s: steady (3 QPS)
    180-210s: spike (20 QPS)
    210-300s: cooldown (1 QPS)
    """
    if t < 60:
        return 2
    elif t < 120:
        return 6
    elif t < 180:
        return 3
    elif t < 210:
        return 20
    else:
        return 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--image", default="image.jpg")
    parser.add_argument("--duration", default=300, type=int)
    parser.add_argument("--output", default="/results/trace.csv")
    args = parser.parse_args()

    arr = preprocess(args.image)

    f = open(args.output, "w")
    f.write("timestamp,latency_ms,qps\n")

    start = time.time()

    while True:
        elapsed = time.time() - start
        if elapsed > args.duration:
            break

        qps = mixed_load_pattern(elapsed)
        interval = 1.0 / qps

        latency = infer(args.url, arr)

        f.write(f"{datetime.utcnow().isoformat()},{latency:.4f},{qps}\n")
        f.flush()

        time.sleep(interval)

    f.close()

if __name__ == "__main__":
    main()
