cat > README.md <<'EOF'
# Native Image Classification (CPU) — ORT gRPC + Metrics

Runs ResNet50 ONNX on ONNX Runtime Server (gRPC), benchmarks latency at fixed
batch sizes, and samples system metrics (CPU%, PSI, and relative energy if RAPL).

*install docker before hand*

## Quickstart

`sudo apt-get install -y python3-venv`
`python3 -m venv .venv`
`source .venv/bin/activate`
`bash ./export_model.sh`
`bash ./simple_run.sh`
