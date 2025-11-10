#!/usr/bin/env bash
set -euo pipefail

# ---- Config (override via env) ----
NS="${NS:-aimodel}"
APP_LABEL="${APP_LABEL:-resnet50}"
SVC="${SVC:-resnet50-service}"
MODEL="${MODEL:-resnet50}"

# Protocol: rest | grpc
PROTO="${PROTO:-rest}"

REST_PORT="${REST_PORT:-8501}"
GRPC_PORT="${GRPC_PORT:-8500}"

REQS="${REQS:-400}"          # total requests (each request sends BATCH images)
CONC="${CONC:-8}"            # concurrency (in-flight requests)
BATCH="${BATCH:-1}"          # images per request (client-side batching)
IMG_URL="${IMG_URL:-https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/kitten_small.jpg}"
OUTDIR="${OUTDIR:-runs}"

# Optional overrides for gRPC if auto-detect fails
SIG="${SIG:-serving_default}"
INPUT_NAME="${INPUT_NAME:-}"

# ---- Run index / timestamp ----
mkdir -p "$OUTDIR"
last="$(ls "$OUTDIR"/bench_*.json 2>/dev/null | sed -E 's/.*bench_([0-9]+)_.*/\1/' | sort -n | tail -1 || true)"
idx="$(printf "%03d" $(( ${last:-0} + 1 )))"
ts="$(date -u +%Y%m%dT%H%M%SZ)"

BENCH_JSON="${OUTDIR}/bench_${idx}_${ts}.json"
LAT_CSV="${OUTDIR}/latency_${idx}_${ts}.csv"
CGROUP_CSV="${OUTDIR}/cgroup_${idx}_${ts}.csv"
ENERGY_CSV="${OUTDIR}/energy_${idx}_${ts}.csv"
TF_BEFORE="${OUTDIR}/tfmetrics_${idx}_${ts}_before.prom"
TF_AFTER="${OUTDIR}/tfmetrics_${idx}_${ts}_after.prom"

echo "Run $idx @ $ts  NS=$NS  APP=$APP_LABEL  SVC=$SVC  MODEL=$MODEL  REQS=$REQS  CONC=$CONC  BATCH=$BATCH  PROTO=$PROTO"

# ---- Locate pod + node ----
POD="$(kubectl -n "$NS" get pod -l app="$APP_LABEL" -o jsonpath='{.items[0].metadata.name}')"
NODE="$(kubectl -n "$NS" get pod "$POD" -o jsonpath='{.spec.nodeName}')"
echo "Using pod: $POD on node: $NODE"

# ---- Port-forward TF-Serving ----
pf_rest_pat="port-forward.*svc/${SVC}.* ${REST_PORT}:${REST_PORT}"
pf_grpc_pat="port-forward.*svc/${SVC}.* ${GRPC_PORT}:${GRPC_PORT}"

pkill -f "$pf_rest_pat" >/dev/null 2>&1 || true
pkill -f "$pf_grpc_pat" >/dev/null 2>&1 || true

kubectl -n "$NS" port-forward "svc/${SVC}" ${REST_PORT}:${REST_PORT} >/dev/null 2>&1 &
PF_REST=$!

if [ "$PROTO" = "grpc" ]; then
  kubectl -n "$NS" port-forward "svc/${SVC}" ${GRPC_PORT}:${GRPC_PORT} >/dev/null 2>&1 &
  PF_GRPC=$!
fi

cleanup() {
  pkill -f "$pf_rest_pat" >/dev/null 2>&1 || true
  pkill -f "$pf_grpc_pat" >/dev/null 2>&1 || true
  kubectl -n "$NS" delete pod rapl-reader-"$idx" --ignore-not-found >/dev/null 2>&1 || true
}
trap cleanup EXIT

# wait until REST is ready (works for both modes)
for i in {1..60}; do
  curl -fs "http://127.0.0.1:${REST_PORT}/v1/models/${MODEL}" >/dev/null && break || sleep 0.2
done

# ---- Grab TF metrics BEFORE (optional) ----
curl -fs "http://127.0.0.1:${REST_PORT}/monitoring/prometheus/metrics" > "$TF_BEFORE" || : > "$TF_BEFORE"

# ---- Prepare CSV headers ----
echo "ts_iso,request_index,latency_ms,status" > "$LAT_CSV"
echo "ts_iso,usage_usec,user_usec,system_usec,usage_delta_ms,cpu_cores_used,cpu_pct_of_one_cpu,psi_some_avg10,psi_some_avg60,psi_some_avg300,psi_some_total,psi_full_avg10,psi_full_avg60,psi_full_avg300,psi_full_total" > "$CGROUP_CSV"
echo "ts_iso,energy_uj,delta_uj,power_w" > "$ENERGY_CSV"

# ---- Minimal privileged helper for RAPL (node energy) ----
cat <<EOF | kubectl -n "$NS" apply -f - >/dev/null
apiVersion: v1
kind: Pod
metadata: { name: rapl-reader-$idx }
spec:
  nodeName: $NODE
  hostPID: true
  hostNetwork: true
  restartPolicy: Never
  containers:
  - name: r
    image: alpine:3.20
    securityContext: { privileged: true }
    command: ["/bin/sh","-lc","sleep 3600"]
    volumeMounts:
      - { name: host, mountPath: /host, readOnly: true }
  volumes:
  - name: host
    hostPath: { path: /, type: Directory }
EOF
kubectl -n "$NS" wait --for=condition=Ready pod/rapl-reader-"$idx" --timeout=30s >/dev/null 2>&1 || true

read_rapl() {
  kubectl -n "$NS" exec rapl-reader-"$idx" -- sh -lc '
    chroot /host /bin/sh -lc "
      p=\$(ls -d /sys/class/powercap/intel-rapl:* 2>/dev/null | head -1);
      if [ -n \"\$p\" ] && [ -r \"\$p/energy_uj\" ]; then cat \$p/energy_uj; else echo NA; fi
    "
  ' 2>/dev/null || echo NA
}

# ---- Start 1Hz sampler (cgroup CPU/PSI + energy) ----
DONE_FLAG="$(mktemp)"; rm -f "$DONE_FLAG"
(
  prev_usage=""
  prev_energy=""
  while [ ! -f "$DONE_FLAG" ]; do
    ts_iso="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    out="$(kubectl -n "$NS" exec "$POD" -- sh -lc '
      {
        cat /sys/fs/cgroup/cpu.stat 2>/dev/null || echo NA;
        echo "---";
        if [ -r /sys/fs/cgroup/cpu.pressure ]; then
          cat /sys/fs/cgroup/cpu.pressure 2>/dev/null || true;
        elif [ -r /proc/pressure/cpu ]; then
          cat /proc/pressure/cpu 2>/dev/null || true;
        else
          echo NA;
        fi
      }
    ' 2>/dev/null || true)"

    cpu="$(echo "$out" | sed -n '1,/^---$/p' | sed '$d')"
    psi="$(echo "$out" | sed -n '/^---$/,$p' | sed '1d')"

    usage="$(echo "$cpu" | awk '/^usage_usec/ {print $2}')"
    user="$(echo "$cpu"  | awk '/^user_usec/  {print $2}')"
    sys="$(echo  "$cpu"  | awk '/^system_usec/{print $2}')"
    [ -z "${usage:-}" ] && usage="NA"

    # deltas & CPU usage
    delta_ms="NA"; cores_used="NA"; pct="NA"
    if [ -n "${prev_usage:-}" ] && [ "$usage" != "NA" ]; then
      d=$((usage - prev_usage))
      [ "$d" -lt 0 ] && d=0
      delta_ms="$(awk -v x="$d" 'BEGIN{printf "%.3f", x/1000.0}')"
      cores_used="$(awk -v ms="$delta_ms" 'BEGIN{printf "%.3f", ms/1000.0}')"     # CPU-seconds per second = cores
      # Clamp 0..100 for "percent of one CPU"
      pct="$(awk -v ms="$delta_ms" 'BEGIN{
        val=(ms/1000.0)*100.0;
        if (val<0) val=0; if (val>100) val=100;
        printf "%.1f", val
      }')"
    fi
    prev_usage="${usage:-0}"

    # PSI parsing (tolerant)
    some_line="$(echo "$psi" | awk '/^some / {print; exit}')"
    full_line="$(echo "$psi" | awk '/^full / {print; exit}')"

    sa10="$(echo "$some_line" | awk -F'[ =]' '{for(i=1;i<=NF;i++) if($i=="avg10"){print $(i+1)}}')"
    sa60="$(echo "$some_line" | awk -F'[ =]' '{for(i=1;i<=NF;i++) if($i=="avg60"){print $(i+1)}}')"
    sa300="$(echo "$some_line"| awk -F'[ =]' '{for(i=1;i<=NF;i++) if($i=="avg300"){print $(i+1)}}')"
    stot="$(echo "$some_line"| awk -F'[ =]' '{for(i=1;i<=NF;i++) if($i=="total"){print $(i+1)}}')"

    fa10="$(echo "$full_line" | awk -F'[ =]' '{for(i=1;i<=NF;i++) if($i=="avg10"){print $(i+1)}}')"
    fa60="$(echo "$full_line" | awk -F'[ =]' '{for(i=1;i<=NF;i++) if($i=="avg60"){print $(i+1)}}')"
    fa300="$(echo "$full_line"| awk -F'[ =]' '{for(i=1;i<=NF;i++) if($i=="avg300"){print $(i+1)}}')"
    ftot="$(echo "$full_line"| awk -F'[ =]' '{for(i=1;i<=NF;i++) if($i=="total"){print $(i+1)}}')"

    echo "$ts_iso,${usage:-NA},${user:-NA},${sys:-NA},$delta_ms,$cores_used,$pct,${sa10:-NA},${sa60:-NA},${sa300:-NA},${stot:-NA},${fa10:-NA},${fa60:-NA},${fa300:-NA},${ftot:-NA}" >> "$CGROUP_CSV"

    # ---- Energy (RAPL) ----
    e="$(read_rapl)"; duj="NA"; pw="NA"
    if [ -n "${e:-}" ] && [ "$e" != "NA" ]; then
      if [ -n "${prev_energy:-}" ]; then
        if [ "$e" -ge "$prev_energy" ] 2>/dev/null; then
          duj=$((e - prev_energy))
          pw="$(awk -v u="$duj" 'BEGIN{printf "%.3f", u/1e6}')"  # J/s = W
        else
          duj="wrap"; pw="NA"
        fi
      fi
      prev_energy="$e"
    fi
    echo "$ts_iso,${e:-NA},$duj,$pw" >> "$ENERGY_CSV"

    sleep 1
  done
) & SAMPLER=$!

# ---- Tiny client venv & load (records per-request CSV) ----
VENV="/tmp/tfserve-csv"
python3 -m venv "$VENV"
# shellcheck disable=SC1090
. "$VENV/bin/activate"

if [ "$PROTO" = "grpc" ]; then
  # grpc + TF Serving API (also installs TF-CPU for TensorProto helpers)
  pip -q install grpcio tensorflow-serving-api==2.17.0 tensorflow-cpu==2.17.0 pillow numpy requests >/dev/null
else
  pip -q install requests pillow numpy >/dev/null
fi

export URL_REST="http://127.0.0.1:${REST_PORT}"
export URL_PRED_REST="${URL_REST}/v1/models/${MODEL}:predict"
export URL_META="${URL_REST}/v1/models/${MODEL}/metadata"
export GRPC_ADDR="127.0.0.1:${GRPC_PORT}"
export MODEL_NAME="${MODEL}"
export SIG_NAME="${SIG}"
export INPUT_NAME_HINT="${INPUT_NAME}"
export IMG="$IMG_URL" N="$REQS" CONC="$CONC" BATCH="$BATCH" OUT_JSON="$BENCH_JSON" OUT_CSV="$LAT_CSV" PROTO_MODE="$PROTO"

python3 - <<'PY'
import io, os, csv, json, time, datetime, requests, numpy as np
from PIL import Image
PROTO=os.environ["PROTO_MODE"]
IMG=os.environ["IMG"]; N=int(os.environ["N"]); CONC=int(os.environ["CONC"]); BATCH=int(os.environ["BATCH"])
OUT_JSON=os.environ["OUT_JSON"]; OUT_CSV=os.environ["OUT_CSV"]
URL_REST=os.environ["URL_REST"]; URL_PRED_REST=os.environ["URL_PRED_REST"]; URL_META=os.environ["URL_META"]
MODEL_NAME=os.environ["MODEL_NAME"]; SIG_NAME=os.environ["SIG_NAME"]; INPUT_NAME_HINT=os.environ["INPUT_NAME_HINT"]

def preprocess(img):
    img = img.convert("RGB").resize((224,224))
    x = np.array(img, dtype=np.float32)
    x = x[..., ::-1]  # RGB->BGR
    x -= np.array([103.939, 116.779, 123.68], dtype=np.float32)
    return x

# Fetch image once
img = Image.open(io.BytesIO(requests.get(IMG, timeout=30).content))
one = preprocess(img)
batch = np.stack([one]*BATCH, axis=0)  # (B, 224,224,3)

def quantiles(lat):
    if not lat: return {}
    s=sorted(lat); L=len(s)
    def q(p): 
        if L==1: return float(s[0])
        k=int(round(p*(L-1))); return float(s[k])
    return {"mean_ms": float(np.mean(s)), "p50_ms": q(0.5), "p90_ms": q(0.9), "p95_ms": q(0.95), "p99_ms": q(0.99)}

if PROTO == "rest":
    payload = {"instances": batch.tolist()}
    import concurrent.futures as cf
    def one_req(i):
        ts_iso = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        t0=time.perf_counter()
        r = requests.post(URL_PRED_REST, json=payload, timeout=60)
        lat_ms=(time.perf_counter()-t0)*1000.0
        return (ts_iso, i, lat_ms, r.status_code)
else:
    import grpc
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
    # Use TensorFlow to build TensorProto efficiently (installed as tensorflow-cpu)
    try:
        import tensorflow as tf
        def to_tensor_proto(np_arr):
            return tf.make_tensor_proto(np_arr, dtype=tf.float32)
    except Exception:
        # Manual fallback (rare)
        from tensorflow.core.framework import tensor_pb2, types_pb2, tensor_shape_pb2
        def to_tensor_proto(np_arr):
            t = tensor_pb2.TensorProto()
            t.dtype = types_pb2.DT_FLOAT
            for s in np_arr.shape:
                t.tensor_shape.dim.add().size = int(s)
            t.tensor_content = np_arr.astype(np.float32).tobytes()
            return t

    # Auto-detect input tensor name from REST metadata (first input of serving_default)
    input_name = INPUT_NAME_HINT.strip() if INPUT_NAME_HINT else ""
    if not input_name:
        try:
            meta = requests.get(URL_META, timeout=10).json()
            sigs = meta.get("metadata",{}).get("signature_def",{}).get("signature_def",{})
            sd = sigs.get(SIG_NAME) or next(iter(sigs.values()))
            inputs = list(sd.get("inputs",{}).keys())
            if inputs: input_name = inputs[0]
        except Exception:
            pass
    if not input_name:
        raise RuntimeError("Could not determine input tensor name. Set INPUT_NAME env to your model's input key.")

    # Prepare static parts
    tensor = to_tensor_proto(batch)
    def build_request():
        req = predict_pb2.PredictRequest()
        req.model_spec.name = MODEL_NAME
        req.model_spec.signature_name = SIG_NAME
        req.inputs[input_name].CopyFrom(tensor)
        return req

    channel = grpc.insecure_channel(os.environ["GRPC_ADDR"])
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    def one_req(i):
        ts_iso = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        req = build_request()
        t0=time.perf_counter()
        try:
            _ = stub.Predict(req, timeout=60)
            code = 0  # OK
        except grpc.RpcError as e:
            code = e.code().value[0] if hasattr(e.code(),"value") else -1
        lat_ms=(time.perf_counter()-t0)*1000.0
        return (ts_iso, i, lat_ms, code)

# Run load
rows=[]
t0=time.time()
from concurrent.futures import ThreadPoolExecutor, as_completed
with ThreadPoolExecutor(max_workers=CONC) as ex:
    futs=[ex.submit(one_req, i+1) for i in range(N)]
    for f in as_completed(futs):
        rows.append(f.result())
t1=time.time()

rows.sort(key=lambda x: x[1])  # by request_index
with open(OUT_CSV, "a", newline="") as f:
    w=csv.writer(f); w.writerows([[a,b,f"{c:.3f}",d] for (a,b,c,d) in rows])

lat = [c for (_,_,c,_) in rows]
stats = quantiles(lat)
out={
  "protocol": PROTO,
  "model": MODEL_NAME,
  "signature": SIG_NAME,
  "input_name": input_name if PROTO=="grpc" else "instances",
  "requests": len(rows),
  "concurrency": CONC,
  "batch_size": BATCH,
  "total_images": len(rows)*BATCH,
  **stats,
  "started_at": int(t0), "finished_at": int(t1), "elapsed_s": t1-t0,
  "rest_url": URL_PRED_REST if PROTO=="rest" else None,
  "grpc_addr": os.environ["GRPC_ADDR"] if PROTO=="grpc" else None
}
with open(OUT_JSON,"w") as f: json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
PY

# ---- Stop sampler; TF metrics AFTER; cleanup ----
touch "$DONE_FLAG"; wait "$SAMPLER" 2>/dev/null || true
curl -fs "http://127.0.0.1:${REST_PORT}/monitoring/prometheus/metrics" > "$TF_AFTER" || : > "$TF_AFTER"

echo
echo "Saved CSVs:"
echo "  $LAT_CSV      # per-request latency (each request carries BATCH images)"
echo "  $CGROUP_CSV   # pod CPU & PSI (1Hz)"
echo "  $ENERGY_CSV   # node energy/power (1Hz, if RAPL available)"
echo "Summary:"
echo "  $BENCH_JSON"
echo "Prometheus text (optional):"
echo "  $TF_BEFORE"
echo "  $TF_AFTER"
