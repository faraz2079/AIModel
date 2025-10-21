

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



Deploy the monitoring stack: 
`git clone --depth 1 https://github.com/prometheus-operator/kube-prometheus; cd kube-prometheus;`

KEPLER_EXPORTER_GRAFANA_DASHBOARD_JSON=`curl -fsSL https://raw.githubusercontent.com/sustainable-computing-io/kepler/main/grafana-dashboards/Kepler-Exporter.json | sed '1 ! s/^/         /'`
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
`until kubectl get servicemonitors --all-namespaces ; do date; sleep 1; echo ""; done
kubectl apply -f manifests/`
