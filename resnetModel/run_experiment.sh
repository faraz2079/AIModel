#!/bin/bash
set -e

# ------------------------
# Config
# ------------------------
NAMESPACE="odm-gpu"
JOB_NAME="resnet50-coop-job"
JOB_YAML="resnet-coop-job-4replicas.yaml"

BASE_DIR="$(pwd)"
RESULTS_BASE="$BASE_DIR/results"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="run_$TIMESTAMP"
RUN_DIR="$RESULTS_BASE/$RUN_ID"

mkdir -p "$RUN_DIR"

# ------------------------
# Extract experiment metadata
# ------------------------
SLICE_ITERS=$(grep -A1 "name: SLICE_ITERS" $JOB_YAML | tail -n1 | tr -d '"' | awk '{print $2}')
TOTAL_ITERS=$(grep -A1 "name: TOTAL_ITERS" $JOB_YAML | tail -n1 | tr -d '"' | awk '{print $2}')
PARALLELISM=$(grep "parallelism:" $JOB_YAML | awk '{print $2}')

# ------------------------
# Experiment descriptor
# ------------------------
cat <<EOF > "$RUN_DIR/experiment.json"
{
  "run_id": "$RUN_ID",
  "job_name": "$JOB_NAME",
  "namespace": "$NAMESPACE",
  "timestamp": "$TIMESTAMP",
  "parallel_jobs": $PARALLELISM,
  "slice_iters": $SLICE_ITERS,
  "total_iters": $TOTAL_ITERS,
  "workload": "ResNet50 forward-pass (TensorFlow, inference-only)",
  "scheduling_policy": "Cooperative GPU time slicing via external scheduler"
}
EOF

echo "Applying job..."

# Inject RUN_ID into Job YAML
sed "s|__RUN_ID__|$RUN_ID|g" "$JOB_YAML" | kubectl apply -f -

# ------------------------
# Measure submit → start delay
# ------------------------
SUBMIT_TIME=$(date +%s.%N)

echo "Waiting for job completion..."
kubectl wait \
  --for=condition=complete \
  job/$JOB_NAME \
  -n $NAMESPACE \
  --timeout=30m

COMPLETION_TIME=$(date +%s.%N)

echo "Job completed."

# ------------------------
# Collector pod (PVC access)
# ------------------------
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
  -n $NAMESPACE \
  --timeout=60s

# ------------------------
# Copy experiment results
# ------------------------
kubectl cp \
  $NAMESPACE/results-collector:/mnt/checkpoints/experiments/$RUN_ID \
  "$RUN_DIR/experiments"

# ------------------------
# Save scheduler logs
# ------------------------
kubectl logs \
  -n default \
  -l app=gpu-scheduler \
  > "$RUN_DIR/scheduler.log"

# ------------------------
# Save timing summary
# ------------------------
cat <<EOF > "$RUN_DIR/timing_summary.json"
{
  "submit_time_unix": $SUBMIT_TIME,
  "completion_time_unix": $COMPLETION_TIME,
  "total_wall_time_sec": $(echo "$COMPLETION_TIME - $SUBMIT_TIME" | bc)
}
EOF

# ------------------------
# Cleanup
# ------------------------
kubectl delete pod results-collector -n $NAMESPACE
kubectl delete job $JOB_NAME -n $NAMESPACE --ignore-not-found

echo "DONE → $RUN_DIR"
