from pathlib import Path

import torch
import open_clip
from PIL import Image

image_path = Path(__file__).parent / "data" / "test.jpg"

if not image_path.is_file():
    raise FileNotFoundError(f"请先放入测试图片：{image_path}")

print("正在加载模型……")
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32-quickgelu",
    pretrained="openai",
    device="cpu",
)
model.eval()

with Image.open(image_path) as image:
    image_tensor = preprocess(image.convert("RGB")).unsqueeze(0)

with torch.inference_mode():
    vector = model.encode_image(image_tensor)
    vector = vector / vector.norm(dim=-1, keepdim=True)

print("图片向量化成功！")
print("向量形状：", tuple(vector.shape))
print("前 8 个数：", vector[0, :8].tolist())