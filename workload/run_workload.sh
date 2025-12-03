#!/bin/bash

JOB_NAME="yolo-loadgen"
NAMESPACE="odm-gpu"
PVC_NAME="odm-loadgen-results"

echo "[1/4] Deleting old job..."
kubectl delete job $JOB_NAME -n $NAMESPACE --ignore-not-found

echo "[2/4] Applying PVC..."
kubectl apply -f pvc-results.yaml

echo "[3/4] Starting new workload job..."
kubectl apply -f loadgen-job.yaml

echo "[4/4] Waiting for job to complete..."
while true; do
    STATUS=$(kubectl get job $JOB_NAME -n $NAMESPACE -o jsonpath='{.status.succeeded}')
    if [ "$STATUS" == "1" ]; then
        echo "Job completed!"
        break
    fi
    echo "Waiting..."
    sleep 5
done

POD=$(kubectl get pod -n $NAMESPACE | grep yolo-loadgen | awk '{print $1}')

echo "Copying trace file..."
kubectl cp -n $NAMESPACE $POD:/results/trace.csv ./trace_yolo.csv 2>/dev/null

echo "Trace saved to trace_yolo.csv"
