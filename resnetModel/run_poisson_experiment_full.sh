#!/bin/bash
set -euo pipefail

# ============================================
# Configuration (ALL OVERRIDABLE)
# ============================================

NAMESPACE="${NAMESPACE:-odm-gpu}"
TEMPLATE="${TEMPLATE:-resnet-poisson-job-template.yaml}"

RESULTS_BASE="${RESULTS_BASE:-$(pwd)/results}"

# ---- Experiment parameters ----
LAMBDA="${LAMBDA:-0.01}"              # jobs/sec
NUM_JOBS="${NUM_JOBS:-40}"

SLICE_ITERS="${SLICE_ITERS:-20}"
TOTAL_ITERS="${TOTAL_ITERS:-200}"
WARMUP_ITERS="${WARMUP_ITERS:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GPU_MEM_LIMIT_MB="${GPU_MEM_LIMIT_MB:-3000}"

# ============================================
# Create run directory
# ============================================

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="run_${TIMESTAMP}"
RUN_DIR="${RESULTS_BASE}/${RUN_ID}"

mkdir -p "$RUN_DIR"

echo "======================================"
echo "Starting Poisson experiment"
echo "RUN_ID: $RUN_ID"
echo "Lambda: $LAMBDA jobs/sec"
echo "Jobs:   $NUM_JOBS"
echo "======================================"

# ============================================
# Save experiment metadata
# ============================================

cat <<EOF > "$RUN_DIR/experiment.json"
{
  "run_id": "$RUN_ID",
  "timestamp": "$TIMESTAMP",
  "arrival_rate_lambda": $LAMBDA,
  "num_jobs": $NUM_JOBS,
  "slice_iters": $SLICE_ITERS,
  "total_iters": $TOTAL_ITERS,
  "warmup_iters": $WARMUP_ITERS,
  "batch_size": $BATCH_SIZE,
  "gpu_mem_limit_mb": $GPU_MEM_LIMIT_MB,
  "workload": "ResNet50 inference",
  "scheduler": "Cooperative GPU time slicing"
}
EOF

# ============================================
# Clean previous jobs
# ============================================

echo "Cleaning previous jobs..."
kubectl delete jobs --all -n "$NAMESPACE" --ignore-not-found=true

# ============================================
# Function: submit job
# ============================================

submit_job() {
    local job_index="$1"
    local job_id
    job_id=$(printf "%04d" "$job_index")

    YAML_FILE="/tmp/job_${RUN_ID}_${job_id}.yaml"

    sed \
        -e "s/__JOB_ID__/${job_id}/g" \
        -e "s/__RUN_ID__/${RUN_ID}/g" \
        -e "s/__SLICE_ITERS__/${SLICE_ITERS}/g" \
        -e "s/__TOTAL_ITERS__/${TOTAL_ITERS}/g" \
        -e "s/__WARMUP_ITERS__/${WARMUP_ITERS}/g" \
        -e "s/__BATCH_SIZE__/${BATCH_SIZE}/g" \
        -e "s/__GPU_MEM_LIMIT_MB__/${GPU_MEM_LIMIT_MB}/g" \
        "$TEMPLATE" > "$YAML_FILE"

    echo "Launching job $job_id"
    kubectl apply -f "$YAML_FILE"
}

# ============================================
# Function: sample exponential interarrival
# ============================================

sample_delay() {
python3 - <<PY
import random, math
lam = float("$LAMBDA")
u = random.random()
dt = -math.log(1.0-u)/lam
print(dt)
PY
}

# ============================================
# Launch jobs with Poisson arrival
# ============================================

echo "Launching jobs with Poisson arrivals..."

START_TS=$(date +%s.%N)

for i in $(seq 1 "$NUM_JOBS"); do

    # Submit job
    submit_job "$i"

    # Sample delay
    delay=$(sample_delay)

    echo "Next job in ${delay}s"

    sleep "$delay"

done

echo "All jobs submitted."

# ============================================
# Wait for jobs to finish
# ============================================

echo "Waiting for jobs to complete..."

while true; do

    DONE=$(kubectl get jobs -n "$NAMESPACE" \
        -l run_id="$RUN_ID" \
        -o jsonpath='{range .items[*]}{.status.succeeded}{"\n"}{end}' \
        | awk '{s+=$1} END {print s+0}')

    echo "Completed: $DONE / $NUM_JOBS"

    if [ "$DONE" -ge "$NUM_JOBS" ]; then
        break
    fi

    sleep 5
done

END_TS=$(date +%s.%N)

echo "All jobs completed."

# ============================================
# Collector pod
# ============================================

echo "Creating collector pod..."

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: results-collector
  namespace: $NAMESPACE
spec:
  restartPolicy: Never
  containers:
  - name: collector
    image: busybox
    command: ["sleep", "600"]
    volumeMounts:
    - name: checkpoints
      mountPath: /mnt/checkpoints
  volumes:
  - name: checkpoints
    persistentVolumeClaim:
      claimName: odm-checkpoints
EOF

kubectl wait \
  --for=condition=Ready \
  pod/results-collector \
  -n "$NAMESPACE" \
  --timeout=60s

# ============================================
# Copy results
# ============================================

echo "Copying experiment data..."

kubectl cp \
  "$NAMESPACE/results-collector:/mnt/checkpoints/experiments/$RUN_ID" \
  "$RUN_DIR/experiments"

# ============================================
# Scheduler logs
# ============================================

echo "Saving scheduler logs..."

kubectl logs \
  -n "$NAMESPACE" \
  -l app=gpu-scheduler \
  > "$RUN_DIR/scheduler.log" || true

# ============================================
# Timing summary
# ============================================

cat <<EOF > "$RUN_DIR/timing_summary.json"
{
  "submit_start_unix": $START_TS,
  "submit_end_unix": $END_TS,
  "total_wall_time_sec": $(python3 - <<PY
s=float("$END_TS"); a=float("$START_TS"); print(s-a)
PY
)
}
EOF

# ============================================
# Cleanup
# ============================================

kubectl delete pod results-collector -n "$NAMESPACE" || true

echo "======================================"
echo "Experiment finished"
echo "Results stored in:"
echo "$RUN_DIR"
echo "======================================"
