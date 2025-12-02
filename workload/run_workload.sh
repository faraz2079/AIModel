#!/bin/bash

JOB_NAME="resnet50-loadgen"
OUTPUT_DIR="./traces"
CONTAINER_NAME="loadgen"

mkdir -p "$OUTPUT_DIR"

echo "[1/4] Deleting old job (if exists)..."
kubectl delete job $JOB_NAME --ignore-not-found=true >/dev/null 2>&1

echo "[2/4] Starting new loadgen job..."
kubectl apply -f loadgen-job.yaml

echo "[3/4] Waiting for pod to start..."
POD=""
while [ -z "$POD" ]; do
    POD=$(kubectl get pods --no-headers | grep "$JOB_NAME" | awk '{print $1}')
    sleep 1
done

echo "Pod detected: $POD"

# Wait until pod enters Running or Completed
PHASE=""
while true; do
    PHASE=$(kubectl get pod $POD -o jsonpath='{.status.phase}')
    echo "Waiting... current status: $PHASE"
    if [[ "$PHASE" == "Running" || "$PHASE" == "Succeeded" || "$PHASE" == "Failed" ]]; then
        break
    fi
    sleep 5
done

echo "Loadgen running... waiting for completion."

# Wait for job completion
while true; do
    STATUS=$(kubectl get job $JOB_NAME -o jsonpath='{.status.succeeded}')
    if [[ "$STATUS" == "1" ]]; then
        echo "Job completed!"
        break
    fi
    echo "Still running..."
    sleep 5
done

echo "[4/4] Copying trace.csv from completed pod..."

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT_FILE="$OUTPUT_DIR/trace_${TIMESTAMP}.csv"

# IMPORTANT: specify container name (required for Completed pods)
kubectl cp $POD:$CONTAINER_NAME:/results/trace.csv $OUTPUT_FILE --retries=5

if [ $? -eq 0 ]; then
    echo "Trace successfully copied to: $OUTPUT_FILE"
else
    echo "ERROR: Failed to copy trace.csv"
    echo "Trying fallback copy from node PVC folder…"
fi

echo "Done!"
