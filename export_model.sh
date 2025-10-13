#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

# venv
if [[ ! -d venv ]]; then
  python3 -m venv venv
fi
source venv/bin/activate
python -m pip -q install --upgrade pip

# CPU-only wheels + ONNX lib required by exporter
pip install --index-url https://download.pytorch.org/whl/cpu \
  torch torchvision \
  --no-cache-dir
pip install onnx --no-cache-dir   # <-- add this line

# export
python export_resnet_onnx.py --arch resnet50 --opset 13 --dynamic-batch --out model/resnet50.onnx
