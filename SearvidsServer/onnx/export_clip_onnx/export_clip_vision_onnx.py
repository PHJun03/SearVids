# export_clip_vision_onnx.py
"""
Export CLIP vision encoder to ONNX.
Requires: transformers, torch, Pillow
"""

import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="openai/clip-vit-base-patch32")
parser.add_argument("--out", default="onnx_models/raw/clip_vision.onnx")
parser.add_argument("--opset", type=int, default=13)
parser.add_argument("--image_size", type=int, default=224)
args = parser.parse_args()

device = "cpu"
print("Loading model and processor...")
model = CLIPModel.from_pretrained(args.model).vision_model.to(device).eval()
processor = CLIPProcessor.from_pretrained(args.model)

# Create dummy image (white)
from PIL import Image
dummy_img = Image.new("RGB", (args.image_size, args.image_size), color=(255,255,255))
inputs = processor(images=dummy_img, return_tensors="pt")
pixel_values = inputs["pixel_values"].to(device)  # shape [1,3,H,W]

print("Exporting to ONNX:", args.out)
os.makedirs(os.path.dirname(args.out), exist_ok=True)

# dynamic axes: batch, height, width (if you want variable spatial dims)
dynamic_axes = {
    "pixel_values": {0: "batch", 2: "height", 3: "width"},
    "last_hidden_state": {0: "batch"}
}

torch.onnx.export(
    model,
    (pixel_values,),
    args.out,
    export_params=True,
    opset_version=args.opset,
    do_constant_folding=True,
    input_names=["pixel_values"],
    output_names=["last_hidden_state"],
    dynamic_axes=dynamic_axes,
)

print("Vision ONNX saved.")
