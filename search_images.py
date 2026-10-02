from pathlib import Path

import torch
import open_clip
from PIL import Image

ROOT = Path(__file__).parent
QUERY = ROOT / "data" / "test.jpg"
LIBRARY = ROOT / "data" / "library"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

if not QUERY.is_file():
    raise FileNotFoundError(f"找不到查询图片：{QUERY}")

files = sorted(
    path for path in LIBRARY.glob("*")
    if path.is_file() and path.suffix.lower() in EXTENSIONS
)
if not files:
    raise RuntimeError(f"素材库里没有图片，请检查：{LIBRARY}")

print("正在加载模型……")
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32-quickgelu",
    pretrained="openai",
    device="cpu",
)
model.eval()


def encode_image(path):
    with Image.open(path) as image:
        tensor = preprocess(image.convert("RGB")).unsqueeze(0)

    with torch.inference_mode():
        vector = model.encode_image(tensor)
        vector = vector / vector.norm(dim=-1, keepdim=True)

    return vector


print("正在处理查询图片……")
query_vector = encode_image(QUERY)

results = []

for index, path in enumerate(files, start=1):
    print(f"处理素材 {index}/{len(files)}：{path.name}")
    try:
        vector = encode_image(path)
        score = (query_vector @ vector.T).item()
        results.append((score, path.name))
    except (OSError, ValueError) as error:
        print(f"跳过无法读取的图片：{path.name}，原因：{error}")

results.sort(key=lambda item: item[0], reverse=True)

if not results:
    raise RuntimeError("没有成功处理的素材图片，请检查图片文件。")

print(f"\n查询图片：{QUERY.name}")
print("最相似的前 5 张图片：")
print("-" * 50)

for rank, (score, name) in enumerate(results[:5], start=1):
    print(f"{rank}. 相似度 {score:.4f}  |  {name}")