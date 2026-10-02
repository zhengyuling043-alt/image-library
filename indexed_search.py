import argparse
import json
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

ROOT = Path(__file__).parent
LIBRARY = ROOT / "data" / "library"
INDEX_DIR = ROOT / "data" / "index"
MODEL_NAME = "ViT-B-32-quickgelu"
WEIGHTS = "openai"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_model():
    print("正在加载模型……")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=WEIGHTS, device="cpu"
    )
    model.eval()
    return model, preprocess


def encode_image(path, model, preprocess):
    with Image.open(path) as image:
        tensor = preprocess(image.convert("RGB")).unsqueeze(0)

    with torch.inference_mode():
        vector = model.encode_image(tensor)
        vector = vector / vector.norm(dim=-1, keepdim=True)

    return vector[0].cpu().numpy()


def build_index():
    files = sorted(
        path for path in LIBRARY.glob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )
    if not files:
        raise RuntimeError(f"素材库没有图片：{LIBRARY}")

    model, preprocess = load_model()
    names, vectors = [], []

    for number, path in enumerate(files, start=1):
        print(f"处理 {number}/{len(files)}：{path.name}")
        try:
            vector = encode_image(path, model, preprocess)
        except (OSError, ValueError) as error:
            print(f"跳过无法读取的图片：{path.name}，原因：{error}")
            continue

        names.append(path.name)
        vectors.append(vector)

    if not vectors:
        raise RuntimeError("没有成功处理的图片，未保存索引。")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / "vectors.npy", np.stack(vectors))

    metadata = {
        "model": MODEL_NAME,
        "weights": WEIGHTS,
        "files": names,
    }
    (INDEX_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n索引建立成功！已保存 {len(names)} 张图片的向量。")


def search(query_path):
    vector_file = INDEX_DIR / "vectors.npy"
    metadata_file = INDEX_DIR / "metadata.json"

    if not vector_file.is_file() or not metadata_file.is_file():
        raise RuntimeError("请先执行：python indexed_search.py build")

    if not query_path.is_file():
        raise FileNotFoundError(f"找不到查询图片：{query_path}")

    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if (
        metadata["model"] != MODEL_NAME
        or metadata["weights"] != WEIGHTS
    ):
        raise RuntimeError("索引使用的模型已变化，请重新建立索引。")

    vectors = np.load(vector_file, allow_pickle=False)
    names = metadata["files"]
    if vectors.ndim != 2 or len(vectors) != len(names):
        raise RuntimeError("索引文件不一致，请重新建立索引。")

    print(f"已读取 {len(names)} 张素材的向量，无需重新编码素材。")
    model, preprocess = load_model()
    query_vector = encode_image(query_path, model, preprocess)
    scores = vectors @ query_vector
    order = np.argsort(-scores)[:5]

    print(f"\n查询图片：{query_path.name}")
    print("最相似的前 5 张图片：")
    for rank, index in enumerate(order, start=1):
        print(f"{rank}. 相似度 {scores[index]:.4f} | {names[index]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["build", "search"])
    parser.add_argument("--query", default="data/test.jpg")
    args = parser.parse_args()

    if args.action == "build":
        build_index()
    else:
        query_path = Path(args.query)
        if not query_path.is_absolute():
            query_path = ROOT / query_path
        search(query_path)