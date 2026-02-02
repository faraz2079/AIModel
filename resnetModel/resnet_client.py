import os
import time
import json
import csv
import logging
from datetime import datetime

import tensorflow as tf
import numpy as np
import requests

# ------------------------
# Environment
# ------------------------
SCHEDULER_URL = os.getenv("SCHEDULER_URL")
SLICE_ITERS = int(os.getenv("SLICE_ITERS", "20"))
TOTAL_ITERS = int(os.getenv("TOTAL_ITERS", "200"))
WARMUP_ITERS = int(os.getenv("WARMUP_ITERS", "5"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
CLIENT_ID = os.getenv("CLIENT_ID", "unknown")
PRIORITY = int(os.getenv("PRIORITY", "0"))
RUN_ID = os.getenv("RUN_ID", "run_unknown")
GPU_MEM_LIMIT_MB = int(os.getenv("GPU_MEM_LIMIT_MB", "3000"))

# MUST match collector
RESULTS_DIR = "/mnt/checkpoints/experiments"

# ------------------------
# Paths
# ------------------------
BASE_DIR = os.path.join(RESULTS_DIR, RUN_ID, "jobs", CLIENT_ID)
os.makedirs(BASE_DIR, exist_ok=True)

LOG_FILE = os.path.join(BASE_DIR, "job.log")
SLICE_CSV = os.path.join(BASE_DIR, "slice_timeline.csv")
METRICS_JSON = os.path.join(BASE_DIR, "metrics.json")

# ------------------------
# Logging
# ------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)

def log(event, **kwargs):
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": event,
        "client_id": CLIENT_ID,
        **kwargs,
    }
    logging.info(json.dumps(payload))

# ------------------------
# TensorFlow GPU setup
# ------------------------
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=GPU_MEM_LIMIT_MB)]
        )
    except RuntimeError:
        pass

def gpu_memory_mb():
    info = tf.config.experimental.get_memory_info("GPU:0")
    return info["current"] / 1024**2, info["peak"] / 1024**2

# ------------------------
# Model
# ------------------------
model = tf.keras.applications.ResNet50(
    weights=None,
    input_shape=(224, 224, 3),
    classes=1000,
)

# ------------------------
# Scheduler API
# ------------------------
def request_gpu():
    requests.post(
        f"{SCHEDULER_URL}/request",
        json={"client_id": CLIENT_ID, "priority": PRIORITY},
        timeout=5,
    )

def release_gpu():
    requests.post(
        f"{SCHEDULER_URL}/release",
        json={"client_id": CLIENT_ID},
        timeout=5,
    )

# ------------------------
# CSV header
# ------------------------
with open(SLICE_CSV, "w", newline="") as f:
    csv.writer(f).writerow([
        "slice_id", "start_iter", "end_iter",
        "slice_time_ms", "gpu_mem_current_mb", "gpu_mem_peak_mb"
    ])

# ------------------------
# Execution
# ------------------------
log(
    "job_start",
    total_iters=TOTAL_ITERS,
    slice_iters=SLICE_ITERS,
    batch_size=BATCH_SIZE,
    gpu_mem_limit_mb=GPU_MEM_LIMIT_MB,
)

iteration = 0
slice_id = 0
slice_times = []
job_start_time = time.time()

while iteration < TOTAL_ITERS:
    slice_id += 1
    start_iter = iteration

    log("slice_request", slice_id=slice_id)
    request_gpu()

    t0 = time.time()

    while iteration < TOTAL_ITERS and iteration < start_iter + SLICE_ITERS:
        if iteration < WARMUP_ITERS:
            iteration += 1
            continue

        x = np.random.rand(BATCH_SIZE, 224, 224, 3).astype(np.float32)
        model(x, training=False)
        iteration += 1

    t1 = time.time()
    release_gpu()

    slice_ms = (t1 - t0) * 1000
    mem_cur, mem_peak = gpu_memory_mb()
    slice_times.append(slice_ms)

    log(
        "slice_complete",
        slice_id=slice_id,
        start_iter=start_iter,
        end_iter=iteration,
        slice_time_ms=slice_ms,
        gpu_mem_current_mb=mem_cur,
        gpu_mem_peak_mb=mem_peak,
    )

    with open(SLICE_CSV, "a", newline="") as f:
        csv.writer(f).writerow([
            slice_id, start_iter, iteration,
            round(slice_ms, 2),
            round(mem_cur, 2),
            round(mem_peak, 2),
        ])

# ------------------------
# Final metrics
# ------------------------
metrics = {
    "client_id": CLIENT_ID,
    "total_iters": TOTAL_ITERS,
    "slice_iters": SLICE_ITERS,
    "num_slices": slice_id,
    "avg_slice_time_ms": sum(slice_times) / len(slice_times),
    "total_runtime_sec": time.time() - job_start_time,
    "gpu_mem_limit_mb": GPU_MEM_LIMIT_MB,
}

with open(METRICS_JSON, "w") as f:
    json.dump(metrics, f, indent=2)

log("job_complete", **metrics)
