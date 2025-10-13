#!/usr/bin/env bash
set -e

# Always remove the ONNX server on exit (or error)
cleanup() {
  echo "[+] Cleaning up Docker container..."
  sudo docker rm -f ortsrv >/dev/null 2>&1 || true
}
trap cleanup EXIT  # <- ensures cleanup() runs when script ends or crashes

echo "[+] Creating virtualenv and installing deps..."
python3 -m venv venv
source venv/bin/activate
pip install -q --upgrade pip
pip install -r requirements.txt

echo "[+] Starting ONNX Runtime server..."
sudo docker run --rm --name ortsrv \
  -p 8000:8000 -p 8001:8001 \
  -v "$(pwd)/model/resnet50.onnx":/models/resnet50/resnet50.onnx:ro \
  mcr.microsoft.com/onnxruntime/server \
  --model_path /models/resnet50/resnet50.onnx \
  --http_port 8000 --grpc_port 8001 \
  --log_level info &
sleep 5

echo "[+] Running benchmark + metrics..."
python measure_grpc_with_metrics.py

echo "[+] Done."
ls -lh metrics/*.csv
