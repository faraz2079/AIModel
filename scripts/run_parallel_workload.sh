#!/usr/bin/env bash
set -euo pipefail

# =====================================================
# Hybrid GPU Inference – Parallel Workload Experiment
#
# NOTE:
# - This script assumes it is run on the GPU node
#   (nvidia-smi is executed locally).
# - For remote execution, prefer DCGM exporter metrics.
# =====================================================

# ==============================
# CONFIG
# ==============================
NAMESPACE="sa"
DEPLOYMENT="hybrid-gpu-infer-longwork"
PORT=8000
WORKLOAD_SECONDS=300
OUTDIR="experiment_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTDIR"

echo "[INFO] Results will be stored in: $OUTDIR"

# ==============================
# CLEANUP HANDLER
# ==============================
cleanup() {
  echo "[INFO] Cleaning up background processes..."
  kill ${PF_PID:-} ${GPU_PID:-} ${LOG_PID:-} 2>/dev/null || true
}
trap cleanup EXIT

# ==============================
# PORT-FORWARD
# ==============================
echo "[INFO] Starting port-forward..."
kubectl port-forward -n "$NAMESPACE" deployment/"$DEPLOYMENT" $PORT:$PORT \
  > "$OUTDIR/portforward.log" 2>&1 &
PF_PID=$!

echo "[INFO] Waiting for port-forward readiness..."
for i in {1..30}; do
  if curl -s "http://localhost:$PORT/docs" >/dev/null 2>&1; then
    echo "[INFO] Port-forward is ready."
    break
  fi
  sleep 1
  if ! kill -0 "$PF_PID" >/dev/null 2>&1; then
    echo "[ERROR] Port-forward process died."
    echo "[ERROR] Check $OUTDIR/portforward.log"
    exit 1
  fi
done

# ==============================
# GPU MONITORING
# ==============================
echo "[INFO] Starting GPU monitoring (nvidia-smi)..."
(
  while true; do
    date "+%F %T"
    nvidia-smi
    echo "-------------------------------------"
    sleep 2
  done
) > "$OUTDIR/nvidia-smi.log" &
GPU_PID=$!

# ==============================
# POD LOGS
# ==============================
echo "[INFO] Streaming pod logs..."
kubectl logs -n "$NAMESPACE" -f deployment/"$DEPLOYMENT" \
  > "$OUTDIR/pod.log" &
LOG_PID=$!

# ==============================
# PREPARE INPUT
# ==============================
echo "[INFO] Preparing test image..."
curl -sL https://ultralytics.com/images/bus.jpg -o "$OUTDIR/test.jpg"
base64 -w 0 "$OUTDIR/test.jpg" > "$OUTDIR/img.b64"

jq -n --rawfile img "$OUTDIR/img.b64" \
  "{task:\"classification\", workload_seconds:$WORKLOAD_SECONDS, image_b64:\$img}" \
  > "$OUTDIR/classify.json"

jq -n --rawfile img "$OUTDIR/img.b64" \
  "{task:\"detection\", workload_seconds:$WORKLOAD_SECONDS, image_b64:\$img}" \
  > "$OUTDIR/detect.json"

# ==============================
# PARALLEL REQUESTS
# ==============================
echo "[INFO] Sending parallel GPU workloads..."

(
  echo "=== CLASSIFICATION START ==="
  date "+%F %T"
  time curl -s -X POST http://localhost:$PORT/infer \
    -H "Content-Type: application/json" \
    --data-binary @"$OUTDIR/classify.json"
  echo
  date "+%F %T"
  echo "=== CLASSIFICATION END ==="
) > "$OUTDIR/classification.out" 2>&1 &

(
  echo "=== DETECTION START ==="
  date "+%F %T"
  time curl -s -X POST http://localhost:$PORT/infer \
    -H "Content-Type: application/json" \
    --data-binary @"$OUTDIR/detect.json"
  echo
  date "+%F %T"
  echo "=== DETECTION END ==="
) > "$OUTDIR/detection.out" 2>&1 &

wait

echo "[INFO] Experiment finished."

# ==============================
# SUMMARY
# ==============================
cat <<EOF

==============================
EXPERIMENT COMPLETE
==============================

Artifacts generated in:
  $OUTDIR/

Key files:
- Pod logs:           pod.log
- GPU utilization:    nvidia-smi.log
- Classification:     classification.out
- Detection:          detection.out

How to analyze:

1) Scheduler behavior / prioritization:
   grep "SCHEDULER" $OUTDIR/pod.log

2) GPU utilization timeline:
   less $OUTDIR/nvidia-smi.log

3) Per-request latency:
   less $OUTDIR/classification.out
   less $OUTDIR/detection.out

==============================
EOF
