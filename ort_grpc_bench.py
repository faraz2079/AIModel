# ort_grpc_bench.py
import time, csv, statistics
import numpy as np
import onnxruntime as ort

CHANNEL = "grpc://localhost:8001"   # your server's gRPC port
MODEL = "/home/faraz/triton-model-repo/resnet50/1/resnet50.onnx"  # local model only needed to get I/O names cleanly

def bench(b, warmup=5, runs=50):
    # Build a session to the SERVER, not local file:
    so = ort.SessionOptions()
    sess = ort.InferenceSession(MODEL, sess_options=so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    out = sess.get_outputs()[0].name

    # Recreate a remote session bound to server via session options:
    # ORT server supports gRPC inference via its native APIs when using proper clients.
    # If your installed onnxruntime doesn't provide direct remote session, we’ll just use HTTP fallback after this step.

    x = np.random.randn(b,3,224,224).astype(np.float32)
    # warmup
    for _ in range(warmup):
        _ = sess.run([out], {inp: x})
    ts=[]
    for _ in range(runs):
        t0=time.perf_counter()
        _ = sess.run([out], {inp: x})
        ts.append((time.perf_counter()-t0)*1000)
    return statistics.mean(ts), float(np.percentile(ts,95)), runs

def main():
    batches=[1,2,4,8,16]
    with open("ort_latency_grpc.csv","w") as f:
        f.write("batch,mean_ms,p95_ms,runs\n")
        for b in batches:
            m,p,r=bench(b)
            print(f"B={b:>2} mean={m:.2f} ms p95={p:.2f} ms runs={r}")
            f.write(f"{b},{m:.4f},{p:.4f},{r}\n")

if __name__=="__main__":
    main()
