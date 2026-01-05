1. Overview

This project implements a hybrid GPU inference service deployed on Kubernetes.
A single pod hosts multiple real ML applications (ResNet-50 for classification and YOLOv8 for object detection) and enforces GPU serialization via an internal scheduler.

The primary goal is to study GPU contention, prioritization, and utilization under parallel long-running workloads, while keeping the deployment realistic and reproducible.

Key features:

	•	Real ML models (ResNet-50, YOLOv8)
	•	Single GPU, single pod
	•	Explicit GPU locking (no concurrent execution)
	•	Parallel request injection
	•	Full experiment logging (scheduler, GPU, pod, results)

2. Application Architecture

API

	•	Framework: FastAPI
	•	Endpoint: POST /infer
	•	Request fields:
	•	task: classification or detection
	•	image_b64: Base64-encoded image
	•	workload_seconds (optional): artificial GPU workload duration

Internal Flow

	1.	Request arrives at /infer
	2.	Image is decoded
	3.	Task is selected (ResNet or YOLO)
	4.	Request enters GPU scheduler
	5.	GPU lock is acquired
	6.	Model runs on GPU
	7.	GPU lock is released
	8.	Result is returned

Only one GPU task runs at a time, even if requests arrive concurrently.

3. Docker Image

Build the Image

From the repository root:

`docker build -t faraz2079/hybrid-gpu-infer:longwork-final -f docker/Dockerfile .`

`docker push faraz2079/hybrid-gpu-infer:longwork-final`

Base image:

	•	pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime


  4. Kubernetes Deployment

Namespace

All resources are deployed in:

`namespace: sa`

Deploy the Long-Workload Version:

`kubectl apply -f k8s/deployment-longwork.yaml`

`kubectl apply -f k8s/service.yaml`

Verify:

`kubectl get pods -n sa`

`kubectl logs -n sa deployment/hybrid-gpu-infer-longwork`

Important settings:

	•	replicas: 1
	•	strategy: Recreate
	•	nvidia.com/gpu: 1

This ensures exclusive GPU ownership.

5. Running the Parallel GPU Experiment

The entire experiment is automated.

One-Command Experiment:

`nohup ./scripts/run_parallel_workload.sh > run.log 2>&1 &`

`tail -f run.log`

For observing: 

`watch -n 2 nvidia-smi`



What the script does:

	1.	Starts port-forwarding to the pod
	2.	Streams pod logs
	3.	Monitors GPU usage (nvidia-smi)
	4.	Downloads a test image
	5.	Sends classification and detection requests in parallel
	6.	Waits for completion
	7.	Stores all artifacts in a timestamped directory

6. Experiment Artifacts

Each run generates:

experiment_YYYYMMDD_HHMMSS/

├── pod.log                 # Full pod + scheduler logs

├── nvidia-smi.log          # GPU utilization timeline

├── classification.out      # Classification response + timing

├── detection.out           # Detection response + timing

├── classify.json

├── detect.json

└── test.jpg

7. How to Analyze Results

Scheduler Prioritization:

`grep "SCHEDULER" experiment_*/pod.log`

You should observe:

	•	Requests queued
	•	GPU lock acquisition
	•	Serialized execution
	•	Completion order

GPU Utilization:

`less experiment_*/nvidia-smi.log`

Expected behavior:

	•	GPU utilization spikes
	•	Only one workload active at a time
	•	Memory usage remains stable

Runtime Comparison:

`less experiment_*/classification.out`

`less experiment_*/detection.out`

Compare:

	•	Start timestamps
	•	End timestamps
	•	Total execution time

9. Intended Use Cases

	•	GPU scheduling research

	•	Kubernetes GPU contention studies

	•	Thesis experiments

	•	Teaching GPU isolation concepts

	•	Baseline for future GPU slicing or MIG comparisons


11. Notes

	•	Do not scale replicas > 1 unless you want multi-pod GPU contention

	•	Do not increase Gunicorn workers for GPU workloads

	•	Always run experiments with deployment-longwork.yaml

	•	Keep experiment folders immutable once generated



