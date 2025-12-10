import torch
import os
from transformers import CLIPModel, CLIPProcessor

# 설정: 원하는 모델 ID (성능을 높이려면 'openai/clip-vit-large-patch14'로 변경 후 C++ 코드 수정 필요)
MODEL_ID = "openai/clip-vit-base-patch32"
OUTPUT_DIR = "../models"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Loading model: {MODEL_ID}...")
model = CLIPModel.from_pretrained(MODEL_ID)
processor = CLIPProcessor.from_pretrained(MODEL_ID)

model.eval()

# -------------------------------------------------------------------
# 1. Text Encoder Export (with Projection)
# -------------------------------------------------------------------
print("Exporting Text Encoder...")

class TextModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids, attention_mask):
        # get_text_features returns the projected (512-dim) embedding
        return self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

text_wrapper = TextModelWrapper(model)

# Dummy input
dummy_text = ["This is a cat."]
text_inputs = processor(text=dummy_text, return_tensors="pt", padding="max_length", max_length=77)
input_ids = text_inputs["input_ids"]
attention_mask = text_inputs["attention_mask"]

torch.onnx.export(
    text_wrapper,
    (input_ids, attention_mask),
    f"{OUTPUT_DIR}/clip_text_sim.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"], # C++ code expects this name, though it's actually pooled output
    dynamic_axes={
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "last_hidden_state": {0: "batch_size"}
    },
    opset_version=14
)

# -------------------------------------------------------------------
# 2. Vision Encoder Export (with Projection)
# -------------------------------------------------------------------
print("Exporting Vision Encoder...")

class VisionModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, pixel_values):
        # get_image_features returns the projected (512-dim) embedding
        return self.model.get_image_features(pixel_values=pixel_values)

vision_wrapper = VisionModelWrapper(model)

# Dummy input
dummy_image = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    vision_wrapper,
    (dummy_image,),
    f"{OUTPUT_DIR}/clip_vision_sim.onnx",
    input_names=["pixel_values"],
    output_names=["last_hidden_state"], # C++ code expects this name
    dynamic_axes={
        "pixel_values": {0: "batch_size"},
        "last_hidden_state": {0: "batch_size"}
    },
    opset_version=14
)

print(f"Done! Models saved to {OUTPUT_DIR}")
print("Text Output Shape:", text_wrapper(input_ids, attention_mask).shape)
print("Vision Output Shape:", vision_wrapper(dummy_image).shape)
