import numpy as np
import requests
from PIL import Image
import json

img = Image.open("kitten_small.jpg")
img = img.resize((224,224)).convert("RGB")
x = np.array(img, dtype=np.float32)

payload = {
    "inputs": [
        {
            "name": "input_tensor",
            "shape": [1, 224, 224, 3],
            "datatype": "FP32",
            "data": x.flatten().tolist()
        }
    ]
}

r = requests.post("http://127.0.0.1:8000/v2/models/resnet50/infer", json=payload)
print(json.dumps(r.json(), indent=2))
