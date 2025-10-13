cat > run_all.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

# 1) venv + deps
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi
source venv/bin/activate
python -m pip -q install --upgrade pip
python -m pip -q install -r requirements.txt

# 2) ensure model exists (export if helper present)
if [ ! -f "model/resnet50.onnx" ]; then
  if [ -f "pytorch/export_resnet_onnx.py" ]; then
    echo "[setup] exporting model to model/resnet50.onnx ..."
    python pytorch/export_resnet_onnx.py model/resnet50.onnx
  else
    echo "[setup] missing model/resnet50.onnx and no export script."
    echo "       copy your ONNX to ./model/resnet50.onnx and re-run."
    exit 1
  fi
fi

# 3) start server in background
echo "[server] starting ONNX Runtime Server..."
( bash ./run_server.sh "$(pwd)/model/resnet50.onnx" ) &
SRV_PID=$!

# 4) wait for gRPC :8001 to accept connections
echo "[server] waiting for gRPC :8001 ..."
for i in {1..60}; do
  (echo > /dev/tcp/127.0.0.1/8001) >/dev/null 2>&1 && break || sleep 1
done

# 5) run metrics+bench
echo "[bench] running measure_grpc_with_metrics.py ..."
python pytorch/measure_grpc_with_metrics.py || true

# 6) stop server
echo "[server] stopping..."
docker rm -f ortsrv >/dev/null 2>&1 || true

echo "[done] outputs:"
echo " - $(pwd)/pytorch/system_metrics.csv"
echo " - $(pwd)/pytorch/ort_latency_grpc.csv"
EOF
chmod +x run_all.sh
