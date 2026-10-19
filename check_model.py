"""Load the deployed OpenCLIP model and encode one real image."""

import argparse
from pathlib import Path

import numpy as np

from indexed_search import LIBRARY, encode_image, load_model


def default_image():
    candidates = sorted(path for path in LIBRARY.iterdir() if path.is_file())
    return candidates[0] if candidates else LIBRARY / "missing.jpg"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=default_image())
    args = parser.parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"测试图片不存在：{args.image}")
    model, preprocess = load_model()
    vector = encode_image(args.image, model, preprocess)
    print("图片向量化成功")
    print("图片：", args.image.resolve())
    print("设备：", next(model.parameters()).device)
    print("向量形状：", vector.shape)
    print("L2范数：", round(float(np.linalg.norm(vector)), 6))
    print("前8维：", vector[:8].tolist())


if __name__ == "__main__":
    main()
