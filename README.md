cat > README.md <<'EOF'
# Native Image Classification (CPU) — ORT gRPC + Metrics

Runs ResNet50 ONNX on ONNX Runtime Server (gRPC), benchmarks latency at fixed
batch sizes, and samples system metrics (CPU%, PSI, and relative energy if RAPL).

## Quickstart
```bash
git clone <this-repo> && cd ai-native-infer-bench
bash run_all.sh
