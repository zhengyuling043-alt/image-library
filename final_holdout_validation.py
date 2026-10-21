"""Generate and evaluate a fixed, never-tuned synthetic final holdout set."""

import io
import json
import random
import shutil
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

from evaluate_duplicates import complete_link_cluster_measure, measure
from group_duplicates import color_distance, hash_distance, inspect_image, phash_distance
from incremental_search import ROOT, fingerprint, read_index, update_index
from make_dataset import BASE_MAKERS
from runtime_compat import LANCZOS


SEED = 20261024
BASES = 24
SIMILARITY = 0.86
DISTANCE = 8
WORK = ROOT / ".runtime" / "final-holdout"


def reset_workdir():
    resolved = WORK.resolve()
    runtime = (ROOT / ".runtime").resolve()
    if resolved.parent != runtime or resolved.name != "final-holdout":
        raise RuntimeError("拒绝清理非预期评测目录")
    if resolved.exists():
        shutil.rmtree(resolved)
    (resolved / "images").mkdir(parents=True)


def save_variants(base, group, folder):
    width, height = base.size
    files = {}
    original = folder / f"{group}_original.png"
    base.save(original)
    files["original"] = original
    copy = folder / f"{group}_copy.png"
    shutil.copyfile(original, copy)
    files["copy"] = copy
    resize = folder / f"{group}_resize.jpg"
    base.resize((width // 2, height // 2), LANCZOS).save(resize, quality=68)
    files["resize"] = resize
    brightness = folder / f"{group}_brightness.jpg"
    ImageEnhance.Brightness(base).enhance(.72).save(brightness, quality=78)
    files["brightness"] = brightness
    crop = folder / f"{group}_crop.png"
    base.crop((int(width * .1), int(height * .1), int(width * .9), int(height * .9))).save(crop)
    files["crop"] = crop
    watermark_image = base.copy()
    draw = ImageDraw.Draw(watermark_image)
    draw.rectangle((int(width * .62), int(height * .82), width, height), fill="white")
    draw.text((int(width * .65), int(height * .86)), "FINAL", fill="black")
    watermark = folder / f"{group}_watermark.png"
    watermark_image.save(watermark)
    files["watermark"] = watermark
    jpeg = folder / f"{group}_jpeg.jpg"
    base.save(jpeg, quality=48)
    files["jpeg"] = jpeg
    return files


def main():
    reset_workdir()
    rng = random.Random(SEED)
    samples = []
    for number in range(BASES):
        width, height = rng.randint(220, 420), rng.randint(220, 420)
        base = BASE_MAKERS[number % len(BASE_MAKERS)](rng, width, height).convert("RGB")
        group = f"holdout_{number:03d}"
        for variant, path in save_variants(base, group, WORK / "images").items():
            samples.append({"name": path.name, "group": group, "variant": variant})

    index = WORK / "index" / "holdout.npz"
    report = update_index(WORK / "images", index, checkpoint_every=50)
    if report["counts"]["failed"]:
        raise RuntimeError("最终留出集存在编码失败")
    records, vectors = read_index(index)
    info = {sample["name"]: inspect_image(WORK / "images" / sample["name"]) for sample in samples}
    rows = []
    for left, right in combinations(samples, 2):
        a, b = left["name"], right["name"]
        rows.append({
            "a": a, "b": b, "positive": left["group"] == right["group"],
            "variant_a": left["variant"], "variant_b": right["variant"],
            "exact": records[a]["sha256"] == records[b]["sha256"],
            "cosine": float(np.clip(vectors[a] @ vectors[b], -1, 1)),
            "distance": hash_distance(info[a], info[b]),
            "phash_distance": phash_distance(info[a], info[b]),
            "color_distance": color_distance(info[a], info[b]),
        })
    pair = measure(rows, SIMILARITY, DISTANCE)
    cluster = complete_link_cluster_measure(rows, SIMILARITY, DISTANCE)
    variants = {}
    for variant in ("copy", "resize", "brightness", "crop", "watermark", "jpeg"):
        subset = [row for row in rows if row["positive"] and
                  {row["variant_a"], row["variant_b"]} == {"original", variant}]
        variants[variant] = measure(subset, SIMILARITY, DISTANCE)
    audit = []
    for similarity in (.86, .88, .90, .92, .94):
        for distance in (6, 8, 12):
            metrics = measure(rows, similarity, distance)
            audit.append({"similarity": similarity, "distance": distance,
                          **{key: value for key, value in metrics.items() if key != "errors"}})
    payload = {
        "scope": "算法与阈值冻结后生成的固定种子合成留出集；未用于调参，不能替代真实用户数据外部验证。",
        "seed": SEED, "base_sources": BASES, "images": len(samples),
        "threshold": {"similarity": SIMILARITY, "distance": DISTANCE},
        "integrity": {"all_index_hashes_current": all(
            fingerprint(WORK / "images" / name) == record["sha256"] for name, record in records.items())},
        "pair_metrics": {key: value for key, value in pair.items() if key != "errors"},
        "cluster_metrics": cluster,
        "original_vs_variant": {name: {key: value for key, value in metrics.items() if key != "errors"}
                                for name, metrics in variants.items()},
        "errors": pair["errors"],
        "posthoc_threshold_audit_not_used_for_selection": audit,
    }
    (ROOT / "FINAL_HOLDOUT_RESULTS.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 最终留出集评测", "", payload["scope"], "",
             f"- 固定随机种子：`{SEED}`；独立原图：{BASES}；变体后共 {len(samples)} 张。",
             f"- 冻结阈值：CLIP相似度 ≥ {SIMILARITY}，多视图dHash距离 ≤ {DISTANCE}。",
             f"- 图片对：Precision {pair['precision']:.2%}，Recall {pair['recall']:.2%}，F1 {pair['f1']:.4f}。",
             f"- 完整组：Precision {cluster['precision']:.2%}，Recall {cluster['recall']:.2%}，F1 {cluster['f1']:.4f}，纯度 {cluster['purity']:.2%}。", "",
             "| 变换 | 检出/应检出 | 召回率 |", "|---|---:|---:|"]
    for name, metrics in variants.items():
        lines.append(f"| {name} | {metrics['tp']} / {metrics['tp'] + metrics['fn']} | {metrics['recall']:.2%} |")
    lines += ["", "详细错误对见 `FINAL_HOLDOUT_RESULTS.json`。评测数据位于忽略提交的 `.runtime/final-holdout`，可由脚本固定种子重建。",
              "命令：`.venv\\Scripts\\python.exe final_holdout_validation.py`。"]
    (ROOT / "FINAL_HOLDOUT_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
