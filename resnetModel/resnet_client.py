import os
import time
import json
import socket
import requests
from datetime import datetime

# NOTE: We intentionally do NOT import tensorflow at module import time.
# We import it only AFTER the scheduler grants a slice to avoid eager GPU init.

def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

SCHEDULER_URL = os.getenv("SCHEDULER_URL", "http://gpu-scheduler.default.svc.cluster.local:8080")
CLIENT_ID = os.getenv("CLIENT_ID", socket.gethostname())
PRIORITY = env_int("PRIORITY", 10)

SLICE_ITERS = env_int("SLICE_ITERS", 20)
TOTAL_ITERS = env_int("TOTAL_ITERS", 200)
WARMUP_ITERS = env_int("WARMUP_ITERS", 10)

BATCH_SIZE = env_int("BATCH_SIZE", 4)

RESULTS_DIR = os.getenv("RESULTS_DIR", "/mnt/checkpoints")
RUN_ID = os.getenv("RUN_ID", "resnet50_real")

# Cooperative memory settings
GPU_MEM_LIMIT_MB = env_int("GPU_MEM_LIMIT_MB", 3000)   # per-pod cap; tune if needed
ALLOW_GROWTH = os.getenv("TF_FORCE_GPU_ALLOW_GROWTH", "true").lower() in ("1", "true", "yes")

POLL_INTERVAL_SEC = env_float("POLL_INTERVAL_SEC", 0.5)
REQUEST_TIMEOUT_SEC = env_float("REQUEST_TIMEOUT_SEC", 2.0)

os.makedirs(RESULTS_DIR, exist_ok=True)

def scheduler_request():
    payload = {
        "client_id": CLIENT_ID,
        "priority": PRIORITY,
        "ts": now_iso(),
    }
    try:
        r = requests.post(f"{SCHEDULER_URL}/request", json=payload, timeout=REQUEST_TIMEOUT_SEC)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"granted": False, "error": str(e)}

def scheduler_release():
    payload = {
        "client_id": CLIENT_ID,
        "ts": now_iso(),
    }
    try:
        r = requests.post(f"{SCHEDULER_URL}/release", json=payload, timeout=REQUEST_TIMEOUT_SEC)
        r.raise_for_status()
        return True
    except Exception:
        return False

def init_tf_and_model():
    """
    Initialize TensorFlow + model only when we are granted a slice.
    This avoids GPU memory allocation during the 'waiting for grant' phase.
    """
    # Set env hints BEFORE importing TF
    if ALLOW_GROWTH:
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    # Optional allocator that can reduce fragmentation in some setups
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("No GPU detected by TensorFlow inside the container.")

    # Configure GPU memory behavior BEFORE any GPU ops
    for gpu in gpus:
        try:
            if ALLOW_GROWTH:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    # Hard cap per process (recommended for multi-tenant)
    # NOTE: This must happen before logical devices are created.
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=GPU_MEM_LIMIT_MB)]
        )
    except Exception:
        # If it fails (e.g., logical devices already initialized), we continue with growth-only.
        pass

    # Build model (weights=None to avoid download; still "real inference" compute)
    model = ResNet50(weights=None)
    return tf, model

def run_inference_iters(tf, model, iters: int, batch_size: int):
    """
    Run iters forward passes. Generate inputs on CPU; TF will place ops on GPU.
    """
    # Generate random images; shape: (B, 224, 224, 3)
    # Use tf.random on CPU to reduce GPU-side random allocation issues.
    with tf.device("/CPU:0"):
        x = tf.random.uniform([batch_size, 224, 224, 3], dtype=tf.float32)

    # Warm-up single call (graph building / kernel selection)
    _ = model(x, training=False)

    t0 = time.time()
    for _i in range(iters):
        _ = model(x, training=False)
    # Force completion
    tf.experimental.async_clear_error() if hasattr(tf.experimental, "async_clear_error") else None
    tf.keras.backend.clear_session()
    t1 = time.time()
    return t1 - t0

def append_jsonl(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")

def main():
    progress = 0
    slice_index = 0

    log_path = os.path.join(RESULTS_DIR, f"{RUN_ID}_{CLIENT_ID}.jsonl")
    append_jsonl(log_path, {"ts": now_iso(), "event": "start", "client_id": CLIENT_ID})

    while progress < TOTAL_ITERS:
        # Wait for grant
        grant = scheduler_request()
        if not grant.get("granted", False):
            time.sleep(POLL_INTERVAL_SEC)
            continue

        # Granted: do one slice
        remaining = TOTAL_ITERS - progress
        this_slice = min(SLICE_ITERS, remaining)

        append_jsonl(log_path, {
            "ts": now_iso(),
            "event": "slice_granted",
            "slice_index": slice_index,
            "progress": progress,
            "slice_iters": this_slice,
            "grant_info": grant,
        })

        # IMPORTANT: Only now initialize TF and model
        try:
            tf, model = init_tf_and_model()
        except Exception as e:
            append_jsonl(log_path, {"ts": now_iso(), "event": "tf_init_error", "error": str(e)})
            scheduler_release()
            raise

        # Warm-up for first slice (optional)
        warmup = WARMUP_ITERS if progress == 0 else 0
        if warmup > 0:
            try:
                _ = run_inference_iters(tf, model, warmup, BATCH_SIZE)
            except Exception as e:
                append_jsonl(log_path, {"ts": now_iso(), "event": "warmup_error", "error": str(e)})
                scheduler_release()
                raise

        # Actual slice execution
        try:
            dt = run_inference_iters(tf, model, this_slice, BATCH_SIZE)
        except Exception as e:
            append_jsonl(log_path, {"ts": now_iso(), "event": "slice_error", "error": str(e)})
            scheduler_release()
            raise

        progress += this_slice

        released = scheduler_release()
        append_jsonl(log_path, {
            "ts": now_iso(),
            "event": "slice_done",
            "slice_index": slice_index,
            "slice_iters": this_slice,
            "progress": progress,
            "slice_seconds": dt,
            "iters_per_sec": (this_slice / dt) if dt > 0 else None,
            "release_ok": released,
        })

        slice_index += 1

    append_jsonl(log_path, {"ts": now_iso(), "event": "completed", "final_progress": progress})
    print(f"[{CLIENT_ID}] Completed. progress={progress}/{TOTAL_ITERS}. Results: {log_path}")

if __name__ == "__main__":
    main()
