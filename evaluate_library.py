"""Evaluate the checked-in 500-image library without re-encoding model vectors.

The source manifest provides group labels and a calibration/test split. Thresholds
are selected only on calibration images, then reported once on the independent
test split. Both naive transitive closure and the product's complete-link
grouping are measured so the anti-chain design is directly auditable.
"""

import argparse
import json
import platform
from itertools import combinations
from pathlib import Path

import numpy as np

from evaluate_duplicates import (
    cluster_measure,
    complete_link_cluster_measure,
    measure,
)
from group_duplicates import color_distance, hash_distance, inspect_image, phash_distance
from incremental_search import EXTENSIONS, INDEX_FILE, LIBRARY, MODEL, ROOT, WEIGHTS, fingerprint, read_index
from vector_index import backend_name


MANIFEST = ROOT / "data" / "photo_manifest.json"
JSON_REPORT = ROOT / "LIBRARY_EVALUATION_RESULTS.json"
MARKDOWN_REPORT = ROOT / "LIBRARY_EVALUATION_RESULTS.md"
SIMILARITIES = (0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98)
DISTANCES = (4, 6, 8, 10, 12, 16, 20)


def load_samples(library=LIBRARY, index_file=INDEX_FILE, manifest_file=MANIFEST):
    manifest = json.loads(Path(manifest_file).read_text(encoding="utf-8"))
    files = {row["name"]: row for row in manifest["files"]}
    split_by_source = {row["source_id"]: row["split"] for row in manifest["groups"]}
    if len(split_by_source) != len(manifest["groups"]):
        raise ValueError("来源分组ID不唯一。")
    records, vectors = read_index(index_file)
    disk_names = {path.name for path in Path(library).iterdir()
                  if path.is_file() and path.suffix.lower() in EXTENSIONS}
    if set(files) != set(records) or set(files) != disk_names:
        raise ValueError("来源清单、素材目录与索引文件名不一致，不能生成可信评测。")

    info, samples = {}, []
    for name in sorted(files):
        path = Path(library) / name
        digest = fingerprint(path)
        if records[name].get("sha256") != digest:
            raise ValueError(f"索引指纹已过期：{name}")
        source_id = files[name]["source_id"]
        if source_id not in split_by_source:
            raise ValueError(f"图片缺少调参/测试分区：{name}")
        info[name] = {**inspect_image(path), "sha256": digest}
        samples.append({**files[name], "split": split_by_source[source_id]})
    return manifest, samples, records, vectors, info


def pair_rows(samples, vectors, info, split):
    selected = [sample for sample in samples if sample["split"] == split]
    rows = []
    for left, right in combinations(selected, 2):
        a, b = left["name"], right["name"]
        rows.append({
            "a": a,
            "b": b,
            "positive": left["source_id"] == right["source_id"],
            "variant_a": left["transform"],
            "variant_b": right["transform"],
            "exact": info[a]["sha256"] == info[b]["sha256"],
            "cosine": float(np.clip(vectors[a] @ vectors[b], -1, 1)),
            "distance": hash_distance(info[a], info[b]),
            "phash_distance": phash_distance(info[a], info[b]),
            "color_distance": color_distance(info[a], info[b]),
        })
    return selected, rows


def choose_threshold(rows):
    trials = []
    for similarity in SIMILARITIES:
        for distance in DISTANCES:
            metrics = measure(rows, similarity, distance)
            trials.append({"similarity": similarity, "distance": distance,
                           **{key: value for key, value in metrics.items() if key != "errors"}})
    eligible = [row for row in trials if row["precision"] >= 0.98]
    if eligible:
        chosen = max(eligible, key=lambda row: (
            row["recall"], row["f1"], row["precision"], row["similarity"], -row["distance"],
        ))
        rule = "仅用调参集；查准率至少98%的候选中优先召回率和F1，仍并列时选择更严格参数"
    else:
        chosen = max(trials, key=lambda row: (
            row["f1"], row["precision"], row["recall"], row["similarity"], -row["distance"],
        ))
        rule = "仅用调参集；没有候选达到98%查准率，因此选择F1最高参数"
    return trials, chosen, rule


def compact_metrics(metrics):
    return {key: value for key, value in metrics.items() if key != "errors"}


def evaluate(library=LIBRARY, index_file=INDEX_FILE, manifest_file=MANIFEST,
             json_report=JSON_REPORT, markdown_report=MARKDOWN_REPORT):
    manifest, samples, records, vectors, info = load_samples(library, index_file, manifest_file)
    calibration_samples, calibration_rows = pair_rows(samples, vectors, info, "calibration")
    test_samples, test_rows = pair_rows(samples, vectors, info, "test")
    trials, chosen, rule = choose_threshold(calibration_rows)
    similarity, distance = chosen["similarity"], chosen["distance"]
    baseline = measure(test_rows, 0.94, 6)
    selected = measure(test_rows, similarity, distance)
    transitive = cluster_measure(test_rows, similarity, distance)
    complete_link = complete_link_cluster_measure(test_rows, similarity, distance)

    variants = sorted({sample["transform"] for sample in test_samples if sample["transform"] != "original"})
    per_variant = {}
    for variant in variants:
        subset = [row for row in test_rows if row["positive"] and
                  {row["variant_a"], row["variant_b"]} == {"original", variant}]
        per_variant[variant] = compact_metrics(measure(subset, similarity, distance))

    result = {
        "scope": "现有500张来源派生素材的回归评测；曾用于定位裁剪问题，不再宣称为最终独立盲测。",
        "assumption": manifest.get("note", "同一source_id视为同组标准答案。"),
        "integrity": {
            "manifest_images": len(manifest["files"]),
            "index_images": len(records),
            "vectors": len(vectors),
            "all_index_hashes_current": True,
        },
        "environment": {
            "python": platform.python_version(),
            "model": MODEL,
            "weights": WEIGHTS,
            "vector_dimension": len(next(iter(vectors.values()))) if vectors else 0,
            "search_backend": backend_name(),
        },
        "splits": {
            "calibration": {"images": len(calibration_samples),
                            "sources": len({row["source_id"] for row in calibration_samples}),
                            "pairs": len(calibration_rows)},
            "test": {"images": len(test_samples),
                     "sources": len({row["source_id"] for row in test_samples}),
                     "pairs": len(test_rows)},
        },
        "selection_rule": rule,
        "calibration_trials": trials,
        "selected_threshold": {"similarity": similarity, "distance": distance},
        "test_default": compact_metrics(baseline),
        "test_selected": compact_metrics(selected),
        "test_errors": selected["errors"],
        "test_cluster_transitive_baseline": transitive,
        "test_cluster_complete_link": complete_link,
        "test_original_vs_variant": per_variant,
    }
    Path(json_report).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    def percent(value):
        return f"{value:.2%}"

    lines = [
        "# 500张素材库重复检测评测", "",
        result["scope"], "", result["assumption"], "",
        "## 数据与环境", "",
        f"- 清单、磁盘和索引一致：{len(records)} 张，索引指纹全部有效。",
        f"- 调参集：{len(calibration_samples)} 张 / {result['splits']['calibration']['sources']} 个来源 / {len(calibration_rows)} 个图片对。",
        f"- 回归测试分区：{len(test_samples)} 张 / {result['splits']['test']['sources']} 个来源 / {len(test_rows)} 个图片对。",
        f"- 模型：`{MODEL}` / `{WEIGHTS}`，{result['environment']['vector_dimension']}维；检索后端：`{result['environment']['search_backend']}`。",
        "", "## 阈值选择", "",
        rule + "。",
        f"选出参数：余弦相似度 `{similarity}`，dHash距离 `≤ {distance}`。", "",
        "| 回归测试分区 | 查准率 | 召回率 | F1 | 误判对 | 漏判对 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| 默认 0.94 / 6 | {percent(baseline['precision'])} | {percent(baseline['recall'])} | {baseline['f1']:.4f} | {baseline['fp']} | {baseline['fn']} |",
        f"| 调参集选出参数 | {percent(selected['precision'])} | {percent(selected['recall'])} | {selected['f1']:.4f} | {selected['fp']} | {selected['fn']} |",
        "", "## 传递性处理对比", "",
        "| 聚类策略 | 查准率 | 召回率 | F1 | 纯度 | 完整组召回率 | 错误合并组 | 被拆分真值组 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| 简单传递闭包（对照） | {percent(transitive['precision'])} | {percent(transitive['recall'])} | {transitive['f1']:.4f} | {percent(transitive['purity'])} | {percent(transitive['whole_group_recall'])} | — | — |",
        f"| 系统全连接约束 | {percent(complete_link['precision'])} | {percent(complete_link['recall'])} | {complete_link['f1']:.4f} | {percent(complete_link['purity'])} | {percent(complete_link['whole_group_recall'])} | {complete_link['overmerge_groups']} | {complete_link['split_truth_groups']} |",
        "", "## 原图对各变换的检出率", "",
        "| 变换 | 找到 / 应找到 | 召回率 |", "|---|---:|---:|",
    ]
    for variant, metrics in per_variant.items():
        expected = metrics["tp"] + metrics["fn"]
        lines.append(f"| {variant} | {metrics['tp']} / {expected} | {percent(metrics['recall'])} |")
    lines += ["", "详细阈值网格、错误图片对和原始数值见 `LIBRARY_EVALUATION_RESULTS.json`。",
              "最终未调参验收见 `FINAL_HOLDOUT_RESULTS.md`；本报告仅用于500张固定素材的回归检查。", ""]
    Path(markdown_report).write_text("\n".join(lines), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=LIBRARY)
    parser.add_argument("--index", type=Path, default=INDEX_FILE)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--json", type=Path, default=JSON_REPORT)
    parser.add_argument("--markdown", type=Path, default=MARKDOWN_REPORT)
    args = parser.parse_args()
    result = evaluate(args.library, args.index, args.manifest, args.json, args.markdown)
    selected = result["test_selected"]
    cluster = result["test_cluster_complete_link"]
    print(f"测试集图片对 P/R/F1：{selected['precision']:.2%} / {selected['recall']:.2%} / {selected['f1']:.4f}")
    print(f"全连接聚类 P/R/F1：{cluster['precision']:.2%} / {cluster['recall']:.2%} / {cluster['f1']:.4f}")
    print(f"报告：{args.markdown.resolve()}")


if __name__ == "__main__":
    main()
