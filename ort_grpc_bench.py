#!/usr/bin/env python3
# ort_grpc_bench.py
import time, csv, statistics
import numpy as np
import onnxruntime as ort
from pathlib import Path

# CHANNEL = "grpc://localhost:8001"  # server's gRPC endpoint
MODEL = str(Path(__file__).resolve().parent / "model" / "resnet50.onnx")  # local ONNX model

def bench(b, warmup=5, runs=50):
    # Load local model just to extract I/O names
    so = ort.SessionOptions()
    sess = ort.InferenceSession(MODEL, sess_options=so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    out = sess.get_outputs()[0].name

    # Prepare dummy input
    x = np.random.randn(b, 3, 224, 224).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        _ = sess.run([out], {inp: x})

    # Timed runs
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = sess.run([out], {inp: x})
        ts.append((time.perf_counter() - t0) * 1000)

    return statistics.mean(ts), float(np.percentile(ts, 95)), runs

def main():
    batches = [1, 2, 4, 8, 16]
    with open("ort_latency_grpc.csv", "w") as f:
        f.write("batch,mean_ms,p95_ms,runs\n")
        for b in batches:
            m, p, r = bench(b)
            print(f"B={b:>2} mean={m:.2f} ms p95={p:.2f} ms runs={r}")
            f.write(f"{b},{m:.4f},{p:.4f},{r}\n")

if __name__ == "__main__":
    main()
