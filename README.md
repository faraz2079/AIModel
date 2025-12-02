This project provides an end-to-end GPU inference pipeline using the NVIDIA Triton Inference Server, a ResNet-50 model, a Python client, and a Kubernetes load generator with full GPU metrics collection (DCGM + Prometheus + Grafana).

The workflow below shows how to deploy this entire system on any new VM from scratch.

1. Clone the Repo

`git clone https://github.com/<your-username>/AIModel.git`

`cd AIModel`

3. Install Dependencies on the VM
   
Nvidia GPU Stack:

`sudo apt update`

`sudo apt install -y nvidia-driver-535 nvidia-container-toolkit`

`sudo nvidia-ctk runtime configure --runtime=crio`

`sudo systemctl restart crio`

check the Driver: 

`nvidia-smi`

3. Deploy Triton Server on Kubernetes

The model is already included in the repo under:

`triton_model_repository/resnet50/`

Deploy Triton:

`kubectl apply -f k8s/triton-gpu.yaml`

`kubectl apply -f k8s/triton-service.yaml`

`kubectl apply -f k8s/triton-prom-scrape.yaml`

4. Test Inference Using Python Client

Activate local Python env:

`source venv/bin/activate`

Run inference:

`python3 client/client_resnet50.py \
  --image scripts/kitten_small.jpg \
  --url http://<NODE-IP>:<NODEPORT>`

5. Run 5-Minute Load Generator (Workload)

`docker pull faraz2079/loadgen:latest`

Run via Kubernetes job:

`cd workload`

`chmod +x run_workload.sh`

`./run_workload.sh`

This script:
	1.	Deletes previous job
	2.	Starts a new 5-minute workload job
	3.	Waits until completion
	4.	Copies trace.csv to the local machine
	5.	Saves it under:

`workload/traces/trace-YYYYMMDD-HHMMSS.csv`

6. Enable GPU Metrics: DCGM Exporter

The project expects DCGM to expose GPU metrics to Prometheus.
Before that make sure you deployed the monitoring stack

`kubectl apply -f https://raw.githubusercontent.com/NVIDIA/dcgm-exporter/master/dcgm-exporter.yaml -n monitoring`

`curl -s localhost:9400/metrics | grep DCGM`

GPU metrics query to visualize in grafana:
	•	DCGM_FI_DEV_GPU_UTIL
  
	•	DCGM_FI_DEV_MEM_COPY_UTIL
  
	•	DCGM_FI_DEV_FB_USED
  
	•	DCGM_FI_DEV_POWER_USAGE
  
	•	DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION
  
	•	DCGM_FI_PROF_PIPE_TENSOR_ACTIVE
  
	•	DCGM_FI_PROF_DRAM_ACTIVE
    
	•	DCGM_FI_PROF_PCIE_RX_BYTES

  	•	DCGM_FI_PROF_PCIE_TX_BYTES
