import torch
from torchvision import models

def export(model_name="resnet50", onnx_path="model.onnx", opset=13):
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    else:
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)

    model.eval()
    # dummy input: NCHW, 1 x 3 x 224 x 224
    x = torch.randn(1, 3, 224, 224)

    # Name inputs/outputs to match Triton config
    input_name = "input__0"
    output_name = "output__0"

    torch.onnx.export(
        model, x, onnx_path,
        input_names=[input_name],
        output_names=[output_name],
        opset_version=opset,
        dynamic_axes={input_name: {0: "batch"}, output_name: {0: "batch"}}
    )
    print(f"Exported {model_name} to {onnx_path}")

if __name__ == "__main__":
    export("resnet50", "model.onnx")  # change to resnet18 if you prefer
