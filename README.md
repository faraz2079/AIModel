# (optional but tidy) python venv
`python3 -m venv ~/venvs/tfexport && source ~/venvs/tfexport/bin/activate`

`pip install --upgrade pip "tensorflow==2.14.1" pillow requests`

# export ResNet50 once; creates SavedModel at $MODEL_DIR/1

`export MODEL_DIR="$HOME/models/resnet50"`

`python - <<'PY'
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
m = ResNet50(weights="imagenet")
tf.saved_model.save(m, f"{__import__('os').environ['MODEL_DIR']}/1")
print("Saved model to:", f"{__import__('os').environ['MODEL_DIR']}/1")
PY`



# Deploy the monitoring stack:

`git clone --depth 1 https://github.com/prometheus-operator/kube-prometheus; cd kube-prometheus;`

KEPLER_EXPORTER_GRAFANA_DASHBOARD_JSON=`curl -fsSL https://raw.githubusercontent.com/sustainable-computing-io/kepler/main/grafana-dashboards/Kepler-Exporter.json | sed '1 ! s/^/ /'`

`mkdir -p grafana-dashboards`

`cat - > ./grafana-dashboards/kepler-exporter-configmap.yaml << EOF
apiVersion: v1
data:
kepler-exporter.json: |-
$KEPLER_EXPORTER_GRAFANA_DASHBOARD_JSON
kind: ConfigMap
metadata:
labels:
app.kubernetes.io/component: grafana
app.kubernetes.io/name: grafana
app.kubernetes.io/part-of: kube-prometheus
app.kubernetes.io/version: 9.5.3
name: grafana-dashboard-kepler-exporter
namespace: monitoring
EOF`


`sudo snap install yq`

`yq -i e '.items += [load("./grafana-dashboards/kepler-exporter-configmap.yaml")]' ./manifests/grafana-dashboardDefinitions.yaml`

`yq -i e '.spec.template.spec.containers.0.volumeMounts += [ {"mountPath": "/grafana-dashboard-definitions/0/kepler-exporter", "name": "grafana-dashboard-kepler-exporter", "readOnly": false} ]' ./manifests/grafana-deployment.yaml`

`yq -i e '.spec.template.spec.volumes += [ {"configMap": {"name": "grafana-dashboard-kepler-exporter"}, "name": "grafana-dashboard-kepler-exporter"} ]' ./manifests/grafana-deployment.yaml`


`kubectl apply --server-side -f manifests/setup`

`until kubectl get servicemonitors --all-namespaces ; do date; sleep 1; echo ""; done`
`kubectl apply -f manifests/`


# For testing the image processing of the model:

> Requires the service to be reachable, e.g.
> `kubectl -n aimodel port-forward svc/resnet50-service 8501:8501`

```bash
# 1) make sure the service is reachable locally
kubectl -n aimodel port-forward svc/resnet50-service 8501:8501 >/dev/null 2>&1 &

# 2) tiny env (no TensorFlow)
python3 -m venv /tmp/tfserve-test && source /tmp/tfserve-test/bin/activate
pip install -q requests pillow numpy

# send N requests and print simple timings
python3 - <<'PY'
import io, time, statistics, numpy as np, requests
from PIL import Image

URL = "http://localhost:8501/v1/models/resnet50:predict"
IMG = "https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/kitten_small.jpg"

def preprocess(img):
img = img.convert("RGB").resize((224,224))
x = np.array(img, dtype=np.float32)
x = x[..., ::-1]
x -= np.array([103.939, 116.779, 123.68], dtype=np.float32)
return np.expand_dims(x, 0).tolist()

img = Image.open(io.BytesIO(requests.get(IMG, timeout=20).content))
x = preprocess(img)

lat = []
for _ in range(200): # bump this if you want more load
t0 = time.perf_counter()
r = requests.post(URL, json={"instances": x}, timeout=30)
r.raise_for_status()
lat.append((time.perf_counter() - t0)*1000)

print(f"Requests: {len(lat)} mean(ms)={statistics.mean(lat):.1f} p95(ms)={statistics.quantiles(lat, n=20)[18]:.1f}")
PY
```


# For running and overriding the defaults:

`./scripts/run_bench_csv.sh`

`REQS=800 CONC=16 NS=aimodel SVC=resnet50-service ./scripts/run_bench_csv.sh`
