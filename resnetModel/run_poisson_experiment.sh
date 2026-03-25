#!/bin/bash
set -euo pipefail

NAMESPACE="odm-gpu"
TEMPLATE="resnet-poisson-job-template.yaml"

# ---- Experiment knobs ----
LAMBDA="${LAMBDA:-0.01}"          # jobs/sec
N_JOBS="${N_JOBS:-20}"            # number of jobs to submit
SLICE_ITERS="${SLICE_ITERS:-20}"
TOTAL_ITERS="${TOTAL_ITERS:-200}"
WARMUP_ITERS="${WARMUP_ITERS:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GPU_MEM_LIMIT_MB="${GPU_MEM_LIMIT_MB:-3000}"

BASE_DIR="$(pwd)"
RESULTS_BASE="$BASE_DIR/results"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="run_poisson_$TIMESTAMP"
RUN_DIR="$RESULTS_BASE/$RUN_ID"
mkdir -p "$RUN_DIR"

echo "RUN_ID=$RUN_ID"
echo "LAMBDA=$LAMBDA jobs/sec, N_JOBS=$N_JOBS"

# record experiment config
cat > "$RUN_DIR/experiment.json" <<EOF
{
  "run_id": "$RUN_ID",
  "namespace": "$NAMESPACE",
  "timestamp": "$TIMESTAMP",
  "arrival_process": "Poisson",
  "lambda_jobs_per_sec": $LAMBDA,
  "n_jobs": $N_JOBS,
  "slice_iters": $SLICE_ITERS,
  "total_iters": $TOTAL_ITERS,
  "warmup_iters": $WARMUP_ITERS,
  "batch_size": $BATCH_SIZE,
  "gpu_mem_limit_mb": $GPU_MEM_LIMIT_MB
}
EOF

# arrival log
ARRIVAL_CSV="$RUN_DIR/arrivals.csv"
echo "job_index,job_id,submit_ts_unix,interarrival_sec" > "$ARRIVAL_CSV"

submit_one_job () {
  local job_index="$1"
  local job_id
  job_id=$(printf "%s-%03d" "$RUN_ID" "$job_index")

  # render template
  sed \
    -e "s/__RUN_ID__/$RUN_ID/g" \
    -e "s/__JOB_ID__/$job_id/g" \
    -e "s/__SLICE_ITERS__/$SLICE_ITERS/g" \
    -e "s/__TOTAL_ITERS__/$TOTAL_ITERS/g" \
    -e "s/__WARMUP_ITERS__/$WARMUP_ITERS/g" \
    -e "s/__BATCH_SIZE__/$BATCH_SIZE/g" \
    -e "s/__GPU_MEM_LIMIT_MB__/$GPU_MEM_LIMIT_MB/g" \
    "$TEMPLATE" | kubectl apply -f -
}

sample_interarrival () {
  python3 - <<PY
import random, math
lam = float("$LAMBDA")
u = random.random()
dt = -math.log(1.0-u)/lam
print(dt)
PY
}

START_TS=$(date +%s.%N)

for i in $(seq 1 "$N_JOBS"); do
  dt=$(sample_interarrival)
  # first job: submit immediately (dt recorded but we don't sleep before first submit)
  if [ "$i" -gt 1 ]; then
    sleep "$dt"
  fi

  now=$(date +%s.%N)
  echo "Submitting job $i/$N_JOBS at $now (dt=$dt)"
  submit_one_job "$i"
  echo "$i,resnet50-poisson-$(printf "%s-%03d" "$RUN_ID" "$i"),$now,$dt" >> "$ARRIVAL_CSV"
done

echo "All jobs submitted. Waiting for completion..."

# wait until all jobs in this run are complete
while true; do
  done_count=$(kubectl get jobs -n "$NAMESPACE" -l run_id="$RUN_ID" -o jsonpath='{range .items[*]}{.status.succeeded}{"\n"}{end}' \
    | awk '{s+=$1} END {print s+0}')
  echo "Completed: $done_count / $N_JOBS"
  if [ "$done_count" -ge "$N_JOBS" ]; then
    break
  fi
  sleep 5
done

END_TS=$(date +%s.%N)

# collector pod (PVC access)
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

kubectl wait --for=condition=Ready pod/results-collector -n "$NAMESPACE" --timeout=60s

# copy experiment results
kubectl cp \
  "$NAMESPACE/results-collector:/mnt/checkpoints/experiments/$RUN_ID" \
  "$RUN_DIR/experiments"

# scheduler logs (NOTE: scheduler is in odm-gpu)
kubectl logs -n odm-gpu -l app=gpu-scheduler > "$RUN_DIR/scheduler.log" || true

cat > "$RUN_DIR/timing_summary.json" <<EOF
{
  "submit_start_unix": $START_TS,
  "submit_end_unix": $END_TS,
  "total_wall_time_sec": $(python3 - <<PY
s=float("$END_TS"); a=float("$START_TS"); print(s-a)
PY
)
}
EOF

kubectl delete pod results-collector -n "$NAMESPACE" || true
echo "DONE -> $RUN_DIR"
