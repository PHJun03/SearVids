import onnxruntime as ort
import numpy as np
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
from PIL import Image

text_onnx = "onnx_models/simplified/clip_text_sim.onnx"
vision_onnx = "onnx_models/simplified/clip_vision_sim.onnx"

# ----------------------
# Load CLIP projection (768 → 512)
# ----------------------
hf_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
proj = hf_model.visual_projection.weight.detach().numpy().T     # shape (768, 512)

# ----------------------
# 텍스트 테스트
# ----------------------
tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
txt = "a photo of a cat"
toks = tokenizer(txt, return_tensors="np", padding="max_length", max_length=64, truncation=True)

sess_text = ort.InferenceSession(text_onnx, providers=ort.get_available_providers())
out = sess_text.run(None, {"input_ids": toks["input_ids"], "attention_mask": toks["attention_mask"]})
text_feats = out[0]   # (1, 64, 512)

# 64 tokens → mean pooling
text_emb = np.mean(text_feats, axis=1)    # (1, 512)
text_emb = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)

print("Text embedding shape:", text_emb.shape)

# ----------------------
# 이미지 테스트
# ----------------------
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
img = Image.new("RGB", (224,224), (255,255,255))
inputs = processor(images=img, return_tensors="np")

sess_vis = ort.InferenceSession(vision_onnx, providers=ort.get_available_providers())
vout = sess_vis.run(None, {"pixel_values": inputs["pixel_values"]})
vis_feats = vout[0]   # (1, 50, 768)

print("Raw vision output shape:", vis_feats.shape)

# CLS(0번째) token 사용
vis_cls = vis_feats[:, 0, :]          # (1, 768)

# projection 적용 (768 → 512)
vis_emb = vis_cls @ proj              # (1, 512)

# normalize
vis_emb = vis_emb / np.linalg.norm(vis_emb, axis=1, keepdims=True)

print("Vision embedding shape:", vis_emb.shape)

# ----------------------
# Cosine similarity
# ----------------------
cos = (text_emb * vis_emb).sum(axis=1)
print("cosine similarity:", cos)
