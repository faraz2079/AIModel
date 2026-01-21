import time
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from collections import deque

app = FastAPI()

# -------------------------
# Scheduler state
# -------------------------
current_owner: Optional[str] = None
queue = deque()

SLICE_SECONDS = 5  # hard time slice

# -------------------------
# Models
# -------------------------
class Request(BaseModel):
    client_id: str
    priority: int = 10


# -------------------------
# Background scheduler loop
# -------------------------
async def scheduler_loop():
    global current_owner

    while True:
        if current_owner is None and queue:
            # Pick highest priority (lower = higher priority)
            queue_list = sorted(list(queue), key=lambda x: (x["priority"], x["ts"]))
            next_client = queue_list.pop(0)

            # Rebuild queue without selected client
            queue.clear()
            queue.extend(queue_list)

            current_owner = next_client["client_id"]
            print(f"[SCHEDULER] Granted GPU to {current_owner}")

            # Enforce slice
            await asyncio.sleep(SLICE_SECONDS)

            print(f"[SCHEDULER] Auto-releasing GPU from {current_owner}")
            current_owner = None

        await asyncio.sleep(0.5)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(scheduler_loop())


# -------------------------
# API endpoints
# -------------------------
@app.post("/request")
def request_gpu(req: Request):
    global current_owner

    if req.client_id == current_owner:
        return {"granted": True}

    queue.append({
        "client_id": req.client_id,
        "priority": req.priority,
        "ts": time.time(),
    })

    return {"granted": False}


@app.post("/release")
def release_gpu(req: Request):
    global current_owner

    if req.client_id == current_owner:
        print(f"[SCHEDULER] Client released GPU: {req.client_id}")
        current_owner = None
        return {"released": True}

    return {"released": False}


@app.get("/state")
def state():
    return {
        "current": current_owner,
        "queue": [
            (item["priority"], item["client_id"])
            for item in queue
        ],
    }
