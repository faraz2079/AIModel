import os
import time
import json
import gc
import requests
import numpy as np

# Reduce TF log noise
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Critical for GPU multi-tenancy: do NOT pre-allocate all memory
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import tensorflow as tf  # must come AFTER env vars

# =========================
# Environment configuration
# =========================
SCHEDULER_URL = os.environ.get("SCHEDULER_URL", "http://gpu-scheduler:8080").rstrip("/")
CLIENT_ID = os.environ.get("CLIENT_ID", "unknown-client")

SLICE_STEPS = int(os.environ.get("SLICE_STEPS", "50"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "200"))
PRIORITY = int(os.environ.get("PRIORITY", "0"))

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/mnt/checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CKPT_PATH = os.path.join(CHECKPOINT_DIR, f"{CLIENT_ID}.json")

# =========================
# Checkpoint helpers
# =========================
def load_progress() -> int:
    if not os.path.exists(CKPT_PATH):
        return 0
    try:
        with open(CKPT_PATH, "r") as f:
            return int(json.load(f).get("current_step", 0))
    except Exception:
        return 0

def save_progress(step: int) -> None:
    tmp = CKPT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(
            {
                "client_id": CLIENT_ID,
                "current_step": step,
                "max_steps": MAX_STEPS,
                "ts": time.time(),
            },
            f,
        )
    os.replace(tmp, CKPT_PATH)

# =========================
# Scheduler API helpers
# =========================
def post_json(path: str, payload: dict, timeout: int = 3):
    return requests.post(f"{SCHEDULER_URL}{path}", json=payload, timeout=timeout)

def request_gpu():
    payloads = [
        {"client_id": CLIENT_ID, "priority": PRIORITY},
        {"client_id": CLIENT_ID},
    ]

    for p in payloads:
        try:
            r = post_json("/request", p)
            if r.status_code != 200:
                continue

            data = r.json()
            granted = False

            if data.get("granted") is True:
                granted = True
            if str(data.get("status", "")).lower() in ("granted", "ok"):
                granted = True
            if data.get("current") == CLIENT_ID:
                granted = True

            return granted, data

        except Exception:
            continue

    return False, {}

def release_gpu():
    try:
        post_json("/release", {"client_id": CLIENT_ID})
        return True
    except Exception:
        return False

def heartbeat():
    try:
        post_json("/heartbeat", {"client_id": CLIENT_ID}, timeout=2)
    except Exception:
        pass

# =========================
# GPU slice execution
# =========================
def do_one_slice(start_step: int, slice_steps: int) -> int:
    model = tf.keras.applications.ResNet50(weights=None)

    batch = int(os.environ.get("BATCH_SIZE", "8"))
    x = tf.constant(
        np.random.rand(batch, 224, 224, 3).astype(np.float32)
    )

    step = start_step
    end_step = min(start_step + slice_steps, MAX_STEPS)

    while step < end_step:
        _ = model(x, training=False)
        step += 1

        if step % 10 == 0:
            heartbeat()

    # Aggressive cleanup (important!)
    del model
    tf.keras.backend.clear_session()
    gc.collect()

    return step

# =========================
# Main loop
# =========================
def main():
    current = load_progress()
    print(
        f"[{CLIENT_ID}] Starting. progress={current}/{MAX_STEPS} priority={PRIORITY}",
        flush=True,
    )

    while current < MAX_STEPS:
        print(f"[{CLIENT_ID}] Waiting for scheduler grant...", flush=True)
        granted, info = request_gpu()

        if not granted:
            time.sleep(1.0)
            continue

        print(
            f"[{CLIENT_ID}] Granted. Running slice: {SLICE_STEPS} steps (from {current}) info={info}",
            flush=True,
        )

        new_step = do_one_slice(current, SLICE_STEPS)
        save_progress(new_step)
        current = new_step

        ok = release_gpu()
        print(
            f"[{CLIENT_ID}] Slice done. progress={current}/{MAX_STEPS}. release_ok={ok}",
            flush=True,
        )

        time.sleep(0.2)

    print(
        f"[{CLIENT_ID}] Completed workload. Final progress={current}/{MAX_STEPS}",
        flush=True,
    )

if __name__ == "__main__":
    main()
