cat > run_server.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH="${1:-$(pwd)/model/resnet50.onnx}"

if [ ! -f "$MODEL_PATH" ]; then
  echo "Model not found at $MODEL_PATH"
  echo "Put resnet50.onnx in ./model or run: python pytorch/export_resnet_onnx.py ./model/resnet50.onnx"
  exit 1
fi

docker rm -f ortsrv >/dev/null 2>&1 || true
docker run --rm --name ortsrv \
  -p 8000:8000 -p 8001:8001 \
  -v "$MODEL_PATH":/models/resnet50/resnet50.onnx:ro \
  mcr.microsoft.com/onnxruntime/server \
  --model_path /models/resnet50/resnet50.onnx \
  --http_port 8000 --grpc_port 8001 --log_level info
EOF
chmod +x run_server.sh
