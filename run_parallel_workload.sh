#!/usr/bin/env bash
set -euo pipefail

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
# PORT-FORWARD
# ==============================
echo "[INFO] Starting port-forward..."
kubectl port-forward -n "$NAMESPACE" deployment/"$DEPLOYMENT" $PORT:$PORT \
  > "$OUTDIR/portforward.log" 2>&1 &
PF_PID=$!

sleep 5

# ==============================
# GPU MONITORING
# ==============================
echo "[INFO] Starting GPU monitoring..."
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
echo "[INFO] Sending parallel requests..."

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

# ==============================
# CLEANUP
# ==============================
echo "[INFO] Stopping background processes..."
kill $PF_PID $GPU_PID $LOG_PID || true

echo "[INFO] Experiment finished."

# ==============================
# SUMMARY
# ==============================
cat <<EOF

==============================
EXPERIMENT COMPLETE
==============================

Artifacts:
- Pod logs:           $OUTDIR/pod.log
- GPU utilization:    $OUTDIR/nvidia-smi.log
- Classification:     $OUTDIR/classification.out
- Detection:          $OUTDIR/detection.out

To verify prioritization:
  grep "SCHEDULER" $OUTDIR/pod.log

To verify GPU usage:
  less $OUTDIR/nvidia-smi.log

==============================
EOF
