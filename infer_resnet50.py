import numpy as np
import requests
import json

# Generate dummy input (1,224,224,3)
input_data = np.zeros((1,224,224,3), dtype=np.float32)

payload = {
    "inputs": [
        {
            "name": "input_1",
            "shape": [1,224,224,3],
            "datatype": "FP32",
            "data": input_data.flatten().tolist()
        }
    ],
    "outputs": [
        {"name": "predictions"}
    ]
}

response = requests.post(
    "http://localhost:8000/v2/models/resnet50/infer",
    headers={"Content-Type": "application/json"},
    data=json.dumps(payload)
)

print("Status:", response.status_code)
print("Response:", response.text[:500], "...")     # show first 500 chars
