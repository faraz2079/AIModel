#!/usr/bin/env python3
# export_resnet_onnx.py
from __future__ import annotations
import argparse
from pathlib import Path

import torch
import torchvision
from torchvision.models import ResNet50_Weights, ResNet18_Weights

ARCHES = {
    "resnet50": (torchvision.models.resnet50, ResNet50_Weights.IMAGENET1K_V1),
    "resnet18": (torchvision.models.resnet18, ResNet18_Weights.IMAGENET1K_V1),
}

def main():
    p = argparse.ArgumentParser(description="Export torchvision ResNet to ONNX.")
    p.add_argument("--arch", choices=ARCHES.keys(), default="resnet50",
                   help="Which model to export (default: resnet50)")
    p.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    p.add_argument("--dynamic-batch", action="store_true",
                   help="Export with dynamic batch dimension")
    p.add_argument("--out", type=Path, default=Path("model/resnet50.onnx"),
                   help="Output ONNX path (default: model/resnet50.onnx)")
    args = p.parse_args()

    # Load model with ImageNet weights
    ctor, weights_enum = ARCHES[args.arch]
    model = ctor(weights=weights_enum)
    model.eval()

    # Dummy input (N, 3, 224, 224)
    dummy = torch.randn(1, 3, 224, 224)

    # Prepare path
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Names & dynamic axes
    input_names = ["input"]
    output_names = ["prob"]
    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {"input": {0: "batch"}, "prob": {0: "batch"}}

    torch.onnx.export(
        model,
        dummy,
        str(args.out),
        input_names=input_names,
        output_names=output_names,
        opset_version=args.opset,
        dynamic_axes=dynamic_axes,
    )

    print(f"[export] wrote {args.out.resolve()} (arch={args.arch}, opset={args.opset}, dynamic_batch={bool(args.dynamic_batch)})")

if __name__ == "__main__":
    main()
