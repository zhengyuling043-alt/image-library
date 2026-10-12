"""近重复分组原型：仅生成待人工核对的结果，不删除图片。"""
import argparse
import json
from itertools import combinations, islice

import numpy as np
from PIL import Image, ImageOps

from incremental_search import LIBRARY, INDEX_FILE, EXTENSIONS, fingerprint, read_index


def inspect_image(path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        width, height = image.size
        gray = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    # 64 位差分感知哈希：相邻像素亮度变化方向。
    bits = (gray[:, 1:] > gray[:, :-1]).ravel()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return {"width": width, "height": height, "dhash": value}


def classify_pair(a, b, info, vectors, similarity, max_distance):
    score = float(np.clip(vectors[a] @ vectors[b], -1, 1))
    distance = (info[a]["dhash"] ^ info[b]["dhash"]).bit_count()
    if info[a]["sha256"] == info[b]["sha256"]:
        category = "exact"
    elif score >= similarity and distance <= max_distance:
        category = "near"
    elif score >= similarity:
        category = "similar"
    else:
        category = "other"
    return {"a": a, "b": b, "category": category,
            "cosine": score, "dhash_distance": distance}


def complete_groups(names, matches):
    """稳定排序后贪心入组；新成员必须与组内每个成员都匹配。"""
    groups = []
    for name in sorted(names):
        for group in groups:
            if all(matches(name, member) for member in group):
                group.append(name)
                break
        else:
            groups.append([name])
    return groups


def generate_groups(library=LIBRARY, index_file=INDEX_FILE, similarity=0.94, max_distance=6, write_report=True, skip_invalid=False):
    if not (-1 <= similarity <= 1) or not (0 <= max_distance <= 64):
        raise ValueError("相似度阈值须在 -1～1，哈希距离须在 0～64。")
    if not library.is_dir() or not index_file.is_file():
        raise RuntimeError("请先准备素材并执行 python incremental_search.py update")
    records, vectors = read_index(index_file)
    info, skipped = {}, {}
    # 索引必须与当前素材一致，避免对过期向量分组。
    disk_names = {p.name for p in library.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS}
    if disk_names != set(records) and not skip_invalid:
        raise RuntimeError("素材与成功入库记录不一致。请先 update 并处理失败图片。")
    for name in disk_names - set(records):
        skipped[name] = "尚未成功建立索引，请更新索引并查看失败原因"
    for name in sorted(records):
        path = library / name
        try:
            digest = fingerprint(path)
            if digest != records[name]["sha256"]:
                raise RuntimeError(f"图片已变化：{name}。请先运行 update。")
            details = {**inspect_image(path), "sha256": digest}
            if fingerprint(path) != digest:
                raise RuntimeError(f"图片在扫描时变化：{name}，请重试。")
            info[name] = details
        except (OSError, ValueError, RuntimeError) as error:
            if not skip_invalid:
                raise
            skipped[name] = str(error)

    # 先按文件哈希形成不可拆分的完全重复组。
    exact_buckets = {}
    for name in sorted(info):
        exact_buckets.setdefault(info[name]["sha256"], []).append(name)
    units = {members[0]: members for members in exact_buckets.values()}
    names = sorted(info)
    positions = {name: i for i, name in enumerate(names)}
    # 分块计算余弦，只保存布尔邻接表，不为数百万图片对创建 Python 字典。
    adjacent = np.eye(len(names), dtype=bool)
    matrix = np.stack([vectors[n] for n in names]) if names else np.empty((0, 512), dtype=np.float32)
    similar_pairs, similar_count = [], 0
    for start in range(0, len(names), 128):
        scores = matrix[start:start + 128] @ matrix.T
        for offset, row in enumerate(scores):
            i = start + offset
            for j in np.flatnonzero(row[i + 1:] >= similarity) + i + 1:
                a, b = names[i], names[j]
                distance = (info[a]["dhash"] ^ info[b]["dhash"]).bit_count()
                if info[a]["sha256"] == info[b]["sha256"] or distance <= max_distance:
                    adjacent[i, j] = adjacent[j, i] = True
                else:
                    similar_count += 1
                    if len(similar_pairs) < 200:
                        similar_pairs.append(classify_pair(a, b, info, vectors, similarity, max_distance))
    # 完全重复判定不依赖向量分数（包括浮点舍入到 1 以下的情况）。
    for members in units.values():
        ids = [positions[name] for name in members]
        adjacent[np.ix_(ids, ids)] = True

    def pair(a, b):
        return classify_pair(a, b, info, vectors, similarity, max_distance)

    # 完全重复组作为整体入组，仍检查不同组之间全部图片对。
    def matches(a, b):
        return all(adjacent[positions[x], positions[y]]
                   for x in units[a] for y in units[b])

    grouped_units = complete_groups(units, matches)
    groups, singles = [], []
    for unit_group in grouped_units:
        members = sorted(name for unit in unit_group for name in units[unit])
        if len(members) == 1:
            singles.extend(members)
            continue
        evidence = [pair(a, b) for a, b in islice(combinations(members, 2), 200)]
        # 仅按像素数量推荐，不声称分辨率最高就一定质量最好。
        keeper = min(members, key=lambda n: (-info[n]["width"] * info[n]["height"], n))
        groups.append({"id": len(groups) + 1,
                       "category": "exact" if len(unit_group) == 1 else "near",
                       "members": members, "suggested_keep": keeper,
                       "keep_reason": "像素数量最多；并列按文件名排序，需人工确认",
                       "pairs": evidence,
                       "pair_count": len(members) * (len(members) - 1) // 2})

    # 输出所有高相似度但未通过近重复规则的图片对，帮助排查误判。
    report = {
        "thresholds": {"cosine": similarity, "dhash_distance": max_distance},
        "notice": "实验阈值，结果待人工核对；不执行删除。相似标签仅指向量达到阈值而哈希未通过。",
        "strategy": "先合并完全重复文件，再按文件名顺序入组，组内所有图片对须通过判定",
        "images": info, "groups": groups, "singletons": singles, "skipped": skipped,
        "similar_pairs_not_merged": similar_pairs,
        "similar_pair_count": similar_count,
    }
    output = index_file.parent / "duplicate_groups.json"
    if write_report:
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not write_report:
        return report
    print(f"共检查 {len(info)} 张图片，发现 {len(groups)} 个候选重复组，{len(singles)} 张未归组。")
    for group in groups:
        label = "完全重复" if group["category"] == "exact" else "含近重复候选"
        print(f"\n组 {group['id']}（{label}）：")
        for name in group["members"]:
            print(f"  {name}")
        print(f"  推荐保留（按像素数）：{group['suggested_keep']}")
        for item in group["pairs"]:
            print(f"  判定 {item['category']} | 相似度 {item['cosine']:.4f} | 哈希距离 {item['dhash_distance']} | {item['a']} / {item['b']}")
    print(f"\n详细结果：{output}\n未删除任何图片。")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--similarity", type=float, default=0.94)
    parser.add_argument("--hash-distance", type=int, default=6)
    args = parser.parse_args()
    generate_groups(similarity=args.similarity, max_distance=args.hash_distance)
