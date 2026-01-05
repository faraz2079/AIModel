#!/usr/bin/env bash
set -euo pipefail

# =====================================================
# Hybrid GPU Inference – Parallel Workload Experiment
#
# WHAT THIS SCRIPT PRODUCES
# - Client end-to-end latency (CSV)
# - Scheduler queue latency (CSV)
# - GPU execution time (CSV)
# - Raw pod logs
# - GPU utilization timeline (nvidia-smi)
#
# ASSUMPTIONS
# - Run on the GPU node (for nvidia-smi)
# - Single replica deployment
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
# CLIENT-SIDE LATENCY CAPTURE
# ==============================
CLIENT_LAT="$OUTDIR/latency_client.csv"
echo "request,type,start_ts,end_ts,latency_seconds" > "$CLIENT_LAT"

run_request () {
  TYPE="$1"
  JSON="$2"

  START=$(date +%s.%N)
  curl -s -X POST http://localhost:$PORT/infer \
    -H "Content-Type: application/json" \
    --data-binary @"$JSON" > "$OUTDIR/${TYPE}.response"
  END=$(date +%s.%N)

  LAT=$(echo "$END - $START" | bc)
  echo "$(date +%F_%T),$TYPE,$START,$END,$LAT" >> "$CLIENT_LAT"
}

echo "[INFO] Sending parallel GPU workloads..."
run_request classification "$OUTDIR/classify.json" &
run_request detection "$OUTDIR/detect.json" &
wait

echo "[INFO] Requests completed."

# ==============================
# POST-PROCESS LATENCIES
# ==============================
echo "[INFO] Extracting scheduler and GPU latencies..."

# Scheduler latency
SCHED_LAT="$OUTDIR/latency_scheduler.csv"
echo "queue_enter_ts,gpu_acquire_ts,wait_seconds" > "$SCHED_LAT"

grep "SCHEDULER" "$OUTDIR/pod.log" \
| awk '
/entered scheduler queue/ { q=$1" "$2 }
/GPU lock acquired/ {
  a=$1" "$2
  cmd="date -d \""q"\" +%s"; cmd | getline qs; close(cmd)
  cmd="date -d \""a"\" +%s"; cmd | getline as; close(cmd)
  print q","a","as-qs
}' >> "$SCHED_LAT"

# GPU execution latency
GPU_LAT="$OUTDIR/latency_gpu.csv"
echo "model,execution_seconds" > "$GPU_LAT"

grep "Inference finished after" "$OUTDIR/pod.log" \
| sed -E 's/.*\[(ResNet|YOLO)\].*after ([0-9.]+)s/\1,\2/' \
>> "$GPU_LAT"

echo "[INFO] Experiment finished."

# ==============================
# SUMMARY
# ==============================
cat <<EOF

==============================
EXPERIMENT COMPLETE
==============================

Artifacts in:
  $OUTDIR/

Core metrics:
- Client latency:        latency_client.csv
- Scheduler wait time:   latency_scheduler.csv
- GPU execution time:    latency_gpu.csv
- GPU utilization:       nvidia-smi.log
- Pod logs:              pod.log

Analysis tips:
- Scheduler behavior:
    grep "SCHEDULER" pod.log
- Plot latency:
    latency_client.csv
- Correlate with energy:
    nvidia-smi.log + DCGM metrics

==============================
EOF
