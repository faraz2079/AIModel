#!/usr/bin/env bash
set -euo pipefail

# ---- Config (override via env) ----
NS="${NS:-aimodel}"
APP_LABEL="${APP_LABEL:-resnet50}"
SVC="${SVC:-resnet50-service}"
MODEL="${MODEL:-resnet50}"
REST_PORT="${REST_PORT:-8501}"
REQS="${REQS:-400}"          # total requests (each request sends BATCH images)
CONC="${CONC:-8}"            # concurrency (in-flight requests)
BATCH="${BATCH:-1}"          # <<< NEW: images per request (client-side batching)
IMG_URL="${IMG_URL:-https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/kitten_small.jpg}"
OUTDIR="${OUTDIR:-runs}"

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

echo "Run $idx @ $ts  NS=$NS  APP=$APP_LABEL  SVC=$SVC  MODEL=$MODEL  REQS=$REQS  CONC=$CONC  BATCH=$BATCH"

# ---- Locate pod + node ----
POD="$(kubectl -n "$NS" get pod -l app="$APP_LABEL" -o jsonpath='{.items[0].metadata.name}')"
NODE="$(kubectl -n "$NS" get pod "$POD" -o jsonpath='{.spec.nodeName}')"
echo "Using pod: $POD on node: $NODE"

# ---- Port-forward TF-Serving ----
pf_pat="port-forward.*svc/${SVC}.* ${REST_PORT}:${REST_PORT}"
pkill -f "$pf_pat" >/dev/null 2>&1 || true
kubectl -n "$NS" port-forward "svc/${SVC}" ${REST_PORT}:${REST_PORT} >/dev/null 2>&1 &
PF_TF=$!

cleanup() {
  pkill -f "$pf_pat" >/dev/null 2>&1 || true
  kubectl -n "$NS" delete pod rapl-reader-"$idx" --ignore-not-found >/dev/null 2>&1 || true
}
trap cleanup EXIT

# wait until ready
for i in {1..60}; do
  curl -fs "http://127.0.0.1:${REST_PORT}/v1/models/${MODEL}" >/dev/null && break || sleep 0.2
done

# ---- Grab TF metrics BEFORE (optional) ----
curl -fs "http://127.0.0.1:${REST_PORT}/monitoring/prometheus/metrics" > "$TF_BEFORE" || : > "$TF_BEFORE"

# ---- Prepare CSV headers ----
echo "ts_iso,request_index,latency_ms,status" > "$LAT_CSV"
echo "ts_iso,usage_usec,user_usec,system_usec,usage_delta_ms,usage_pct_of_one_cpu,psi_some_avg10,psi_some_avg60,psi_some_avg300,psi_some_total,psi_full_avg10,psi_full_avg60,psi_full_avg300,psi_full_total" > "$CGROUP_CSV"
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
    # cpu.stat + psi from inside pod
    out="$(kubectl -n "$NS" exec "$POD" -- sh -lc '
      { cat /sys/fs/cgroup/cpu.stat 2>/dev/null || echo NA; echo "---"; cat /sys/fs/cgroup/cpu.pressure 2>/dev/null || echo NA; }' 2>/dev/null || true)"
    cpu="$(echo "$out" | sed -n '1,/^---$/p' | sed '$d')"
    psi="$(echo "$out" | sed -n '/^---$/,$p' | sed '1d')"

    usage=$(echo "$cpu" | awk "/^usage_usec/ {print \$2}")
    user=$(echo "$cpu"  | awk "/^user_usec/ {print \$2}")
    sys=$(echo "$cpu"   | awk "/^system_usec/ {print \$2}")
    [ -z "$usage" ] && usage="NA"

    # deltas & % of one CPU
    delta_ms="NA"; pct="NA"
    if [ -n "${prev_usage:-}" ] && [ "$usage" != "NA" ]; then
      d=$((usage - prev_usage))
      [ "$d" -lt 0 ] && d=0
      delta_ms="$(awk -v x="$d" 'BEGIN{printf "%.3f", x/1000.0}')"  # usec -> ms
      pct="$(awk -v ms="$delta_ms" 'BEGIN{printf "%.1f", (ms/1000.0)*100.0}')" # ~% of one CPU
    fi
    prev_usage="${usage:-0}"

    # parse PSI (some/full lines)
    some_line="$(echo "$psi" | awk '/^some / {print}')"
    full_line="$(echo "$full_line" | cat -)" # placeholder to avoid unbound; will set properly next:
    full_line="$(echo "$psi" | awk '/^full / {print}')"
    sa10="$(echo "$some_line" | awk -F'[ =]' '{for(i=1;i<=NF;i++)if($i=="avg10")print $(i+1)}')"
    sa60="$(echo "$some_line" | awk -F'[ =]' '{for(i=1;i<=NF;i++)if($i=="avg60")print $(i+1)}')"
    sa300="$(echo "$some_line"| awk -F'[ =]' '{for(i=1;i<=NF;i++)if($i=="avg300")print $(i+1)}')"
    stot="$(echo "$some_line"| awk -F'[ =]' '{for(i=1;i<=NF;i++)if($i=="total")print $(i+1)}')"
    fa10="$(echo "$full_line" | awk -F'[ =]' '{for(i=1;i<=NF;i++)if($i=="avg10")print $(i+1)}')"
    fa60="$(echo "$full_line" | awk -F'[ =]' '{for(i=1;i<=NF;i++)if($i=="avg60")print $(i+1)}')"
    fa300="$(echo "$full_line"| awk -F'[ =]' '{for(i=1;i<=NF;i++)if($i=="avg300")print $(i+1)}')"
    ftot="$(echo "$full_line"| awk -F'[ =]' '{for(i=1;i<=NF;i++)if($i=="total")print $(i+1)}')"

    echo "$ts_iso,${usage:-NA},${user:-NA},${sys:-NA},$delta_ms,$pct,${sa10:-NA},${sa60:-NA},${sa300:-NA},${stot:-NA},${fa10:-NA},${fa60:-NA},${fa300:-NA},${ftot:-NA}" >> "$CGROUP_CSV"

    # energy
    e="$(read_rapl)"; duj="NA"; pw="NA"
    if [ -n "$e" ] && [ "$e" != "NA" ]; then
      if [ -n "${prev_energy:-}" ]; then
        if [ "$e" -ge "$prev_energy" ]; then
          duj=$((e - prev_energy))
          pw="$(awk -v u="$duj" 'BEGIN{printf "%.3f", u/1e6}')"
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
pip -q install requests pillow numpy >/dev/null

URL="http://127.0.0.1:${REST_PORT}/v1/models/${MODEL}:predict" \
IMG="$IMG_URL" N="$REQS" CONC="$CONC" BATCH="$BATCH" OUT_JSON="$BENCH_JSON" OUT_CSV="$LAT_CSV" \
python3 - <<'PY'
import io, os, csv, json, time, datetime, requests, numpy as np
from PIL import Image
URL=os.environ["URL"]; IMG=os.environ["IMG"]; N=int(os.environ["N"]); CONC=int(os.environ["CONC"]); BATCH=int(os.environ["BATCH"])
OUT_JSON=os.environ["OUT_JSON"]; OUT_CSV=os.environ["OUT_CSV"]

def preprocess(img):
    img = img.convert("RGB").resize((224,224))
    x = np.array(img, dtype=np.float32); x = x[..., ::-1]
    x -= np.array([103.939, 116.779, 123.68], dtype=np.float32)
    return x  # single image (no batch dim)

# Build batched payload (replicate same image BATCH times)
img = Image.open(io.BytesIO(requests.get(IMG, timeout=30).content))
one = preprocess(img).tolist()
instances = [one for _ in range(BATCH)]
payload = {"instances": instances}

from concurrent.futures import ThreadPoolExecutor, as_completed
def one_req(i):
    ts_iso = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    t0=time.perf_counter()
    r = requests.post(URL, json=payload, timeout=60)
    lat_ms=(time.perf_counter()-t0)*1000
    return (ts_iso, i, lat_ms, r.status_code)

rows=[]
t0=time.time()
with ThreadPoolExecutor(max_workers=CONC) as ex:
    futs=[ex.submit(one_req, i+1) for i in range(N)]
    for f in as_completed(futs): rows.append(f.result())
t1=time.time()

rows.sort(key=lambda x: x[1])  # by request_index
with open(OUT_CSV, "a", newline="") as f:
    w=csv.writer(f); w.writerows([[a,b,f"{c:.3f}",d] for (a,b,c,d) in rows])

lat = [c for (_,_,c,_) in rows]
lat.sort()
def q(p): 
    if not lat: return None
    k=int(round(p*(len(lat)-1))); return float(lat[k])

out={
  "requests": len(rows),
  "concurrency": CONC,
  "batch_size": BATCH,
  "total_images": len(rows)*BATCH,
  "mean_ms": float(np.mean(lat)) if lat else None,
  "p50_ms": q(0.5), "p90_ms": q(0.9), "p95_ms": q(0.95), "p99_ms": q(0.99),
  "started_at": int(t0), "finished_at": int(t1), "elapsed_s": t1-t0, "url": URL
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
