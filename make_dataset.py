"""程序化生成带同组标注的合成素材，扩充素材库容量并提供重复检测真值。

每张"原图"随机属于四类视觉模式之一（渐变、几何图形、条纹棋盘、平滑噪声），
按固定概率派生完全重复副本（字节相同）或 2~3 个近重复变体（缩放、重压缩、
裁剪、亮度变化），其余为独立单张。同组标注写入标签 JSON，可直接用于
重复检测的准确率/召回率评测与库容量演示。

用法：
    python make_dataset.py                    # 默认 1500 张原图，约 3300 张素材
    python make_dataset.py --bases 200        # 小规模试跑
    python make_dataset.py --seed 7           # 复现同一次生成
"""
import argparse
import io
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from runtime_compat import BILINEAR, LANCZOS

ROOT = Path(__file__).parent
LIBRARY = ROOT / "data/library"
LABELS = ROOT / "data/synthetic_labels.json"

# 近重复变体类型：与 group_duplicates 的 near 档口径对应
VARIANTS = ("resize", "jpeg", "crop", "brightness")


def gradient(rng, width, height):
    """双色线性渐变，方向随机。"""
    top = np.array([rng.randrange(256) for _ in range(3)], dtype=np.float32)
    bottom = np.array([rng.randrange(256) for _ in range(3)], dtype=np.float32)
    if rng.random() < 0.5:
        ramp = np.linspace(0, 1, width, dtype=np.float32)[None, :, None]
        arr = top[None, None, :] * (1 - ramp) + bottom[None, None, :] * ramp
        arr = np.broadcast_to(arr, (height, width, 3))
    else:
        ramp = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
        arr = top[None, None, :] * (1 - ramp) + bottom[None, None, :] * ramp
        arr = np.broadcast_to(arr, (height, width, 3))
    return Image.fromarray(arr.astype(np.uint8))


def shapes(rng, width, height):
    """纯色背景上随机几何图形。"""
    background = tuple(rng.randrange(256) for _ in range(3))
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    for _ in range(rng.randint(3, 8)):
        color = tuple(rng.randrange(256) for _ in range(3))
        x1, y1 = rng.randrange(width), rng.randrange(height)
        x2, y2 = x1 + rng.randrange(width // 4, width), y1 + rng.randrange(height // 4, height)
        if rng.random() < 0.5:
            draw.ellipse((x1, y1, x2, y2), fill=color)
        else:
            draw.rectangle((x1, y1, x2, y2), fill=color)
    return image


def stripes(rng, width, height):
    """条纹或棋盘格。"""
    color_a = tuple(rng.randrange(256) for _ in range(3))
    color_b = tuple(rng.randrange(256) for _ in range(3))
    period = rng.randint(6, max(7, width // 6))
    checker = rng.random() < 0.5
    rows = np.arange(height)[:, None] // period
    cols = np.arange(width)[None, :] // period
    if checker:
        mask = (rows + cols) % 2 == 0
    elif rng.random() < 0.5:
        mask = rows % 2 == 0
    else:
        mask = cols % 2 == 0
    mask = np.broadcast_to(mask, (height, width))
    arr = np.where(mask[..., None], np.array(color_a), np.array(color_b)).astype(np.uint8)
    return Image.fromarray(arr)


def smooth_noise(rng, width, height):
    """小分辨率随机噪声放大 + 轻微模糊，得到平滑色块图。"""
    rows, cols = rng.randint(4, 12), rng.randint(4, 12)
    small = [[[int(rng.random() * 255) for _ in range(3)] for _ in range(cols)] for _ in range(rows)]
    image = Image.fromarray(np.array(small, dtype=np.uint8))
    image = image.resize((width, height), BILINEAR)
    return image.filter(ImageFilter.GaussianBlur(radius=1))


BASE_MAKERS = (gradient, shapes, stripes, smooth_noise)


def render_variant(base, kind, rng):
    """按类型生成变体并返回 (文件字节, 扩展名)。"""
    image = base
    width, height = base.size
    if kind == "resize":
        scale = rng.uniform(0.45, 0.75)
        image = base.resize((max(8, int(width * scale)), max(8, int(height * scale))),
                            LANCZOS)
    elif kind == "crop":
        dx, dy = int(width * rng.uniform(0.03, 0.06)), int(height * rng.uniform(0.03, 0.06))
        image = base.crop((dx, dy, width - dx, height - dy))
    elif kind == "brightness":
        arr = np.asarray(base, dtype=np.float32) * rng.uniform(0.75, 1.25)
        image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buffer = io.BytesIO()
    if kind == "crop":
        image.save(buffer, "PNG")
        return buffer.getvalue(), ".png"
    image.save(buffer, "JPEG", quality=rng.randint(55, 85))
    return buffer.getvalue(), ".jpg"


def generate(bases, seed, library, labels_path):
    if bases < 1:
        raise ValueError("原图数量必须大于 0")
    if labels_path.exists() or any(library.glob("syn_*.jpg")) or any(library.glob("syn_*.png")):
        raise ValueError("目标中已有合成图片或标注；请用 --library 和 --labels 指定新位置，避免覆盖旧数据。")
    rng = random.Random(seed)
    library.mkdir(parents=True, exist_ok=True)
    groups, singletons, total = [], [], 0
    for number in range(1, bases + 1):
        width = rng.randint(160, 320)
        height = rng.randint(160, 320)
        base = BASE_MAKERS[number % len(BASE_MAKERS)](rng, width, height)
        stem = f"syn_g{number:04d}"
        buffer = io.BytesIO()
        base.save(buffer, "JPEG", quality=90)
        base_bytes = buffer.getvalue()
        extension = ".jpg"
        roll = rng.random()
        if roll < 0.35:
            members = [f"{stem}_base{extension}", f"{stem}_copy{extension}"]
            kind = "exact"
            for name in members:
                (library / name).write_bytes(base_bytes)
        elif roll < 0.70:
            picked = rng.sample(VARIANTS, k=rng.randint(2, 3))
            members = [f"{stem}_base{extension}"]
            (library / members[0]).write_bytes(base_bytes)
            for index, kind_of_variant in enumerate(picked, 1):
                data, suffix = render_variant(base, kind_of_variant, rng)
                name = f"{stem}_v{index}_{kind_of_variant}{suffix}"
                (library / name).write_bytes(data)
                members.append(name)
            kind = "near"
        else:
            members = []
            kind = None
        if members:
            groups.append({"id": stem, "kind": kind, "members": members})
            total += len(members)
        else:
            name = f"syn_s{number:04d}.jpg"
            (library / name).write_bytes(base_bytes)
            singletons.append(name)
            total += 1
        if number % 300 == 0:
            print(f"已生成 {number}/{bases} 张原图的素材…")
    labels = {
        "version": 1,
        "seed": seed,
        "bases": bases,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "groups": groups,
        "singletons": singletons,
        "note": ("程序化生成的合成素材：groups 即生成时的真值分组（exact=字节相同，"
                 "near=同源变体），singletons 为独立单张。可直接用于重复检测的"
                 "准确率/召回率评测；不同原图在视觉上也可能偶然相似，评测结果反映"
                 "真实检出能力而非生成意图。"),
    }
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n生成完成：{total} 张素材（{len(groups)} 个重复组 + {len(singletons)} 张单张）。")
    print(f"同组标注已写入：{labels_path}")
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bases", type=int, default=1500, help="原图数量（默认 1500，约生成 3300 张素材）")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子，同种子可复现")
    parser.add_argument("--library", type=Path, default=LIBRARY, help="素材输出目录")
    parser.add_argument("--labels", type=Path, default=LABELS, help="同组标注 JSON 输出路径")
    arguments = parser.parse_args()
    generate(arguments.bases, arguments.seed, arguments.library, arguments.labels)


if __name__ == "__main__":
    main()
