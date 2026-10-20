"""生成带来源标注的独立测试集，评测图片对的重复判定。"""
import argparse
import json
import random
import shutil
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

from group_duplicates import complete_groups, inspect_image
from incremental_search import (
    ROOT, LIBRARY, EXTENSIONS, fingerprint, read_index, update_index,
)
from runtime_compat import LANCZOS, popcount

DEFAULT_DATASET = ROOT / "data" / "evaluation"
VARIANTS = ("original", "copy", "resize", "brightness", "crop", "watermark", "jpeg")


def write_json(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare(source=LIBRARY, destination=DEFAULT_DATASET, count=10, seed=42, pattern="pexels-*"):
    if count < 4:
        raise ValueError("至少使用 4 张不同原图，确保两个分区都有正负样本。")
    if destination.exists():
        raise RuntimeError(f"目录已存在，为避免覆盖请使用新的 --dataset 路径：{destination}")
    candidates = sorted(p for p in source.glob(pattern) if p.is_file() and p.suffix.lower() in EXTENSIONS)
    random.Random(seed).shuffle(candidates)
    selected, digests, skipped = [], set(), []
    for path in candidates:
        try:
            digest = fingerprint(path)
            if digest in digests:
                continue
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError) as error:
            skipped.append({"file": path.name, "error": str(error)})
            continue
        selected.append((path, digest))
        digests.add(digest)
        if len(selected) == count:
            break
    if len(selected) < count:
        raise RuntimeError(f"仅找到 {len(selected)} 张可读且文件哈希不同的原图，需要 {count} 张。")

    calibration_count = max(2, min(count - 2, round(count * .6)))
    samples, sources = [], []
    destination.mkdir(parents=True)
    for number, (path, digest) in enumerate(selected, 1):
        source_id = f"source_{number:03d}"
        split = "calibration" if number <= calibration_count else "test"
        folder = destination / split / "images"
        folder.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as raw:
            base = ImageOps.exif_transpose(raw).convert("RGB")
            base.thumbnail((1024, 1024), LANCZOS)
        width, height = base.size
        original = folder / f"{source_id}_original.png"
        base.save(original)
        shutil.copyfile(original, folder / f"{source_id}_copy.png")
        base.resize((max(1, width // 2), max(1, height // 2)), LANCZOS).save(folder / f"{source_id}_resize.png")
        ImageEnhance.Brightness(base).enhance(.7).save(folder / f"{source_id}_brightness.png")
        dx, dy = int(width * .1), int(height * .1)
        base.crop((dx, dy, width - dx, height - dy)).save(folder / f"{source_id}_crop.png")
        watermark = base.copy()
        draw = ImageDraw.Draw(watermark)
        draw.rectangle((int(width * .65), int(height * .82), width - 1, height - 1), fill="white")
        draw.text((int(width * .67), int(height * .86)), "SAMPLE", fill="black")
        watermark.save(folder / f"{source_id}_watermark.png")
        base.save(folder / f"{source_id}_jpeg.jpg", quality=55)
        if fingerprint(path) != digest:
            raise RuntimeError(f"生成期间原图发生变化：{path}；请使用新目录重新准备。")
        sources.append({"id": source_id, "source_file": str(path.resolve()),
                        "sha256": digest, "split": split})
        for variant in VARIANTS:
            suffix = ".jpg" if variant == "jpeg" else ".png"
            generated = folder / f"{source_id}_{variant}{suffix}"
            samples.append({"file": generated.name, "source_id": source_id,
                            "variant": variant, "split": split, "sha256": fingerprint(generated)})
        print(f"已生成 {source_id} 的 7 张测试图（{split}）：{path.name}")
    manifest = {
        "version": 1, "seed": seed, "sources": sources, "samples": samples,
        "skipped_sources": skipped,
        "assumption": "每个 source_id 视为独立原图；仅排除完全相同文件，需人工检查不同来源是否其实是近重复。",
        "transforms": {"base_max_size": 1024, "resize": .5, "brightness": .7,
                       "crop_each_edge": .1, "watermark": "右下角白底 SAMPLE", "jpeg_quality": 55},
    }
    write_json(destination / "manifest.json", manifest)
    print(f"\n生成完成：{len(samples)} 张图片。调参原图 {calibration_count} 张，测试原图 {count - calibration_count} 张。")
    return manifest


def validate_dataset(dataset, manifest):
    source_splits, seen_files = {}, set()
    for sample in manifest["samples"]:
        split, name = sample["split"], sample["file"]
        if split not in {"calibration", "test"} or Path(name).name != name:
            raise ValueError("无效的测试集文件记录。")
        if (split, name) in seen_files:
            raise ValueError("测试集包含重复文件记录。")
        seen_files.add((split, name))
        source_splits.setdefault(sample["source_id"], set()).add(split)
        if fingerprint(dataset / split / "images" / name) != sample["sha256"]:
            raise RuntimeError(f"评测图片已被修改：{name}，不能继续沿用原始标注。")
    if any(len(splits) != 1 for splits in source_splits.values()):
        raise ValueError("同一原图的版本出现在两个分区，存在数据泄漏。")
    for split in ("calibration", "test"):
        subset = [s for s in manifest["samples"] if s["split"] == split]
        if len({s["source_id"] for s in subset}) < 2:
            raise ValueError("每个分区至少需要两个原图来源。")
        names = {p.name for p in (dataset / split / "images").iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS}
        if names != {s["file"] for s in subset}:
            raise ValueError("测试目录中图片与来源记录不一致。")


def measure(rows, similarity, distance):
    counts = dict(tp=0, fp=0, fn=0, tn=0)
    errors = []
    for row in rows:
        predicted = row["exact"] or (row["cosine"] >= similarity and row["distance"] <= distance)
        key = ("tp" if row["positive"] else "fp") if predicted else ("fn" if row["positive"] else "tn")
        counts[key] += 1
        if key in {"fp", "fn"}:
            errors.append({**row, "error": key})
    tp, fp, fn, tn = (counts[k] for k in ("tp", "fp", "fn", "tn"))
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    return {**counts, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.,
            "false_positive_rate": fp / (fp + tn) if fp + tn else 0., "errors": errors}


def cluster_measure(rows, similarity, distance):
    """Evaluate the transitive grouping induced by pair decisions."""

    names = sorted({row[key] for row in rows for key in ("a", "b")})

    def clusters(links):
        parent = {name: name for name in names}
        def find(name):
            while parent[name] != name:
                parent[name] = parent[parent[name]]
                name = parent[name]
            return name
        for a, b in links:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        result = {}
        for name in names:
            result.setdefault(find(name), []).append(name)
        return list(result.values())

    truth_links = [(row["a"], row["b"]) for row in rows if row["positive"]]
    predicted_links = [
        (row["a"], row["b"]) for row in rows
        if row["exact"] or (row["cosine"] >= similarity and row["distance"] <= distance)
    ]
    truth, predicted = clusters(truth_links), clusters(predicted_links)
    truth_id = {name: index for index, group in enumerate(truth) for name in group}
    predicted_id = {name: index for index, group in enumerate(predicted) for name in group}
    pair_rows_for_groups = []
    for index, a in enumerate(names):
        for b in names[index + 1:]:
            pair_rows_for_groups.append({
                "positive": truth_id[a] == truth_id[b],
                "predicted": predicted_id[a] == predicted_id[b],
            })
    tp = sum(row["positive"] and row["predicted"] for row in pair_rows_for_groups)
    fp = sum(not row["positive"] and row["predicted"] for row in pair_rows_for_groups)
    fn = sum(row["positive"] and not row["predicted"] for row in pair_rows_for_groups)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    purity_hits = sum(max((sum(truth_id[name] == label for name in group)
                           for label in set(truth_id.values())), default=0)
                      for group in predicted)
    complete_truth_groups = sum(len({predicted_id[name] for name in group}) == 1 for group in truth)
    return {"predicted_groups": len(predicted), "truth_groups": len(truth),
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "purity": purity_hits / len(names) if names else 0.0,
            "whole_group_recall": complete_truth_groups / len(truth) if truth else 0.0}


def complete_link_cluster_measure(rows, similarity, distance):
    """Evaluate the deterministic complete-link grouping used by the product.

    Exact duplicate files are first kept as indivisible units. A unit may join a
    predicted group only when every cross-unit image pair passes the decision
    rule. This mirrors :func:`group_duplicates.generate_groups` and prevents a
    transitive A-B-C chain from merging A with C.
    """

    names = sorted({row[key] for row in rows for key in ("a", "b")})
    if not names:
        return {"predicted_groups": 0, "truth_groups": 0, "precision": 0.0,
                "recall": 0.0, "f1": 0.0, "purity": 0.0,
                "whole_group_recall": 0.0, "overmerge_groups": 0,
                "split_truth_groups": 0}
    lookup = {frozenset((row["a"], row["b"])): row for row in rows}

    def connected_groups(links):
        parent = {name: name for name in names}

        def find(name):
            while parent[name] != name:
                parent[name] = parent[parent[name]]
                name = parent[name]
            return name

        for a, b in links:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        grouped = {}
        for name in names:
            grouped.setdefault(find(name), []).append(name)
        return [sorted(group) for group in grouped.values()]

    truth = connected_groups((row["a"], row["b"]) for row in rows if row["positive"])
    exact_units = connected_groups((row["a"], row["b"]) for row in rows if row["exact"])
    units = {group[0]: group for group in sorted(exact_units, key=lambda group: group[0])}

    def predicted_pair(a, b):
        row = lookup[frozenset((a, b))]
        return row["exact"] or (row["cosine"] >= similarity and row["distance"] <= distance)

    def units_match(a, b):
        return all(predicted_pair(left, right) for left in units[a] for right in units[b])

    predicted = [sorted(name for unit in group for name in units[unit])
                 for group in complete_groups(units, units_match)]
    truth_id = {name: index for index, group in enumerate(truth) for name in group}
    predicted_id = {name: index for index, group in enumerate(predicted) for name in group}
    tp = fp = fn = 0
    for index, a in enumerate(names):
        for b in names[index + 1:]:
            positive = truth_id[a] == truth_id[b]
            same_group = predicted_id[a] == predicted_id[b]
            tp += positive and same_group
            fp += not positive and same_group
            fn += positive and not same_group
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    purity_hits = sum(max((sum(truth_id[name] == label for name in group)
                           for label in set(truth_id.values())), default=0)
                      for group in predicted)
    complete_truth_groups = sum(len({predicted_id[name] for name in group}) == 1 for group in truth)
    overmerge_groups = sum(len({truth_id[name] for name in group}) > 1 for group in predicted)
    split_truth_groups = sum(len({predicted_id[name] for name in group}) > 1 for group in truth)
    return {"predicted_groups": len(predicted), "truth_groups": len(truth),
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "purity": purity_hits / len(names),
            "whole_group_recall": complete_truth_groups / len(truth),
            "overmerge_groups": overmerge_groups,
            "split_truth_groups": split_truth_groups}


def pair_rows(dataset, split, samples):
    folder = dataset / split
    report = update_index(folder / "images", folder / "index" / "incremental.npz")
    if report["counts"]["failed"]:
        raise RuntimeError("存在编码失败图片，停止评测，避免把缺失样本排除后虚增指标。")
    records, vectors = read_index(folder / "index" / "incremental.npz")
    info = {s["file"]: inspect_image(folder / "images" / s["file"]) for s in samples}
    rows = []
    for a, b in combinations(samples, 2):
        na, nb = a["file"], b["file"]
        rows.append({"a": na, "b": nb, "positive": a["source_id"] == b["source_id"],
                     "variant_a": a["variant"], "variant_b": b["variant"],
                     "exact": records[na]["sha256"] == records[nb]["sha256"],
                     "cosine": float(np.clip(vectors[na] @ vectors[nb], -1, 1)),
                     "distance": popcount(info[na]["dhash"] ^ info[nb]["dhash"])})
    return rows


def evaluate(dataset=DEFAULT_DATASET):
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    validate_dataset(dataset, manifest)
    calibration = pair_rows(dataset, "calibration", [s for s in manifest["samples"] if s["split"] == "calibration"])
    trials = []
    for similarity in (.90, .94, .97):
        for distance in (6, 12, 18):
            metrics = measure(calibration, similarity, distance)
            trials.append({"similarity": similarity, "distance": distance,
                           **{k: v for k, v in metrics.items() if k != "errors"}})
    # 仅依照调参集选择；优先查准率 >= 98%，再选召回率最高者。
    eligible = [t for t in trials if t["precision"] >= .98]
    if eligible:
        chosen = max(eligible, key=lambda t: (t["recall"], t["precision"], t["similarity"], -t["distance"]))
        selection = "调参集查准率至少 98% 的候选中召回率最高；并列优先更严格阈值"
    else:
        chosen = max(trials, key=lambda t: (t["f1"], t["precision"], t["similarity"], -t["distance"]))
        selection = "没有候选达到查准率 98%，仅报告调参集 F1 最高者，不建议直接用于清理"
    # 阈值选择完成后才处理独立测试分区。
    test = pair_rows(dataset, "test", [s for s in manifest["samples"] if s["split"] == "test"])
    baseline = measure(test, .94, 6)
    selected = measure(test, chosen["similarity"], chosen["distance"])
    cluster_selected = cluster_measure(test, chosen["similarity"], chosen["distance"])
    per_variant = {}
    for variant in VARIANTS[1:]:
        subset = [r for r in test if r["positive"] and {r["variant_a"], r["variant_b"]} == {"original", variant}]
        per_variant[variant] = measure(subset, chosen["similarity"], chosen["distance"])
    validate_dataset(dataset, manifest)
    result = {"scope": "图片对判定评测，不等同于最终分组评测；小样本合成测试，不能代表真实部署精度。",
              "selection_rule": selection, "calibration_trials": trials,
              "selected_threshold": {"similarity": chosen["similarity"], "distance": chosen["distance"]},
              "test_baseline": baseline, "test_selected": selected,
              "test_cluster_selected": cluster_selected,
              "test_original_vs_variant": per_variant,
              "assumption": manifest["assumption"]}
    write_json(dataset / "evaluation_results.json", result)
    lines = ["# 重复判定初步评测", "", result["scope"], "", manifest["assumption"], "",
             f"原图数：{len(manifest['sources'])}；图片总数：{len(manifest['samples'])}。",
             "同一原图的所有版本只属于同一分区；不同分区之间不比较。", "",
             f"候选阈值：相似度 {chosen['similarity']}，哈希距离 {chosen['distance']}。", selection, "",
             "| 独立测试集 | 查准率 | 召回率 | F1 | 误判对数 | 漏判对数 |",
             "|---|---:|---:|---:|---:|---:|"]
    for label, values in (("默认阈值 0.94 / 6", baseline), ("调参集选出阈值", selected)):
        lines.append(f"| {label} | {values['precision']:.2%} | {values['recall']:.2%} | {values['f1']:.4f} | {values['fp']} | {values['fn']} |")
    lines += ["", "## 聚类级指标（包含传递闭包影响）", "",
              "| 分组查准率 | 分组召回率 | 分组 F1 | 纯度 | 完整组召回率 |",
              "|---:|---:|---:|---:|---:|",
              (f"| {cluster_selected['precision']:.2%} | {cluster_selected['recall']:.2%} | "
               f"{cluster_selected['f1']:.4f} | {cluster_selected['purity']:.2%} | "
               f"{cluster_selected['whole_group_recall']:.2%} |")]
    lines += ["", "## 原图与各变换版本的检出情况（选出阈值）", "", "| 变换 | 找到 / 应找到 |", "|---|---:|"]
    for variant, metrics in per_variant.items():
        lines.append(f"| {variant} | {metrics['tp']} / {metrics['tp'] + metrics['fn']} |")
    lines += ["", "具体误判、漏判文件名及分数见 evaluation_results.json。",
              "本工具不修改分组程序默认阈值。若反复根据测试集修改算法，该测试集将不再是独立验收数据，应另准备新原图。"]
    (dataset / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "evaluate"])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--source", type=Path, default=LIBRARY)
    parser.add_argument("--pattern", default="pexels-*")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.source, args.dataset, args.count, args.seed, args.pattern)
    else:
        evaluate(args.dataset)
