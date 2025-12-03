import requests
import time
import csv
import argparse

def load_image(path):
    with open(path, "rb") as f:
        return f.read()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--url", required=True, help="Inference URL, e.g. http://IP:PORT/infer")
    parser.add_argument("--duration", type=int, default=300, help="Duration in seconds (default 300 = 5 min)")
    args = parser.parse_args()

    img_bytes = load_image(args.image)

    start = time.time()
    results = []

    print("Starting 5-minute workload...")

    while time.time() - start < args.duration:
        t0 = time.time()
        try:
            r = requests.post(args.url, files={"file": ("img.jpg", img_bytes, "image/jpeg")})
            latency = (time.time() - t0) * 1000  # ms
            results.append(latency)

        except Exception as e:
            results.append(-1)

    print("Workload completed. Total requests:", len(results))

    with open("/results/trace.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["latency_ms"])
        for x in results:
            writer.writerow([x])

    print("Trace saved to /results/trace.csv")

if __name__ == "__main__":
    main()
