# export_clip_text_onnx.py
"""
Export CLIP text encoder to ONNX.
Requires: transformers, torch
"""

import torch
from transformers import CLIPModel, CLIPTokenizerFast
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="openai/clip-vit-base-patch32")
parser.add_argument("--out", default="onnx_models/raw/clip_text.onnx")
parser.add_argument("--opset", type=int, default=13)
parser.add_argument("--max_seq_length", type=int, default=32)
args = parser.parse_args()

device = "cpu"
print("Loading model and tokenizer...")
model = CLIPModel.from_pretrained(args.model).text_model.to(device).eval()
tokenizer = CLIPTokenizerFast.from_pretrained(args.model)

# Dummy text
dummy_text = "This is a test"
tokens = tokenizer(dummy_text, return_tensors="pt", padding="max_length",
                   truncation=True, max_length=args.max_seq_length)

input_ids = tokens["input_ids"].to(device)
attention_mask = tokens["attention_mask"].to(device)

# The CLIP text model forward signature: input_ids, attention_mask
print("Exporting to ONNX:", args.out)
os.makedirs(os.path.dirname(args.out), exist_ok=True)

# dynamic axes: batch and sequence
dynamic_axes = {
    "input_ids": {0: "batch", 1: "seq"},
    "attention_mask": {0: "batch", 1: "seq"},
    "last_hidden_state": {0: "batch", 1: "seq"}
}

torch.onnx.export(
    model,
    (input_ids, attention_mask),
    args.out,
    export_params=True,
    opset_version=args.opset,
    do_constant_folding=True,
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],
    dynamic_axes=dynamic_axes,
)

print("Text ONNX saved.")