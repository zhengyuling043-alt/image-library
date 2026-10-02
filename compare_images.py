from pathlib import Path

import torch
import open_clip
from PIL import Image

data_dir = Path(__file__).parent / "data"
names = ["test.jpg", "test_similar.jpg", "test_other.jpg"]

for name in names:
    if not (data_dir / name).is_file():
        raise FileNotFoundError(f"找不到图片：{data_dir / name}")

print("正在加载模型……")
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32-quickgelu",
    pretrained="openai",
    device="cpu",
)
model.eval()

images = []
for name in names:
    with Image.open(data_dir / name) as image:
        images.append(preprocess(image.convert("RGB")))

with torch.inference_mode():
    vectors = model.encode_image(torch.stack(images))
    vectors = vectors / vectors.norm(dim=-1, keepdim=True)
    scores = vectors @ vectors.T

print(f"原图 vs 原图自身：{scores[0, 0].item():.4f}")
print(f"原图 vs 修改版本：{scores[0, 1].item():.4f}")
print(f"原图 vs 不同图片：{scores[0, 2].item():.4f}")