from fastapi import FastAPI, HTTPException
import time

app = FastAPI()

current_owner = None

@app.post("/acquire")
def acquire(client_id: str):
    global current_owner
    if current_owner is not None:
        raise HTTPException(status_code=409, detail="GPU busy")
    current_owner = client_id
    return {"status": "granted", "owner": current_owner}

@app.post("/release")
def release(client_id: str):
    global current_owner
    if current_owner != client_id:
        raise HTTPException(status_code=403, detail="Not owner")
    current_owner = None
    return {"status": "released"}

@app.get("/state")
def state():
    return {"current_owner": current_owner}
