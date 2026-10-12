"""
近重复图片检测与分组模块。

本模块用于在图片库中检测完全重复和近似重复的图片，并将它们分组，
供人工审核决定是否删除冗余文件。模块不会自动删除任何图片。

工作原理：
    采用三层相似度判定策略，逐步细化：
    1. 精确哈希匹配（SHA-256）：通过文件内容的 SHA-256 摘要判断两张图片
       是否完全相同（字节级一致）。这是最严格的判定，结果不可拆分。
    2. 感知哈希（dHash）：将图片转换为 9x8 灰度图，比较相邻像素的亮度
       变化方向，生成 64 位差分哈希。两张图片的 dHash 汉明距离越小，
       说明视觉结构越相似。默认阈值为 6 位以内。
    3. CLIP 向量相似度：使用预训练的 CLIP 模型将图片编码为 512 维向量，
       通过余弦相似度衡量语义层面的相似程度。默认阈值为 0.94。

分组算法：
    - 首先按 SHA-256 将完全相同的文件合并为"单元组"（exact buckets）。
    - 然后分块计算所有图片对的 CLIP 余弦相似度，结合 dHash 距离构建
      布尔邻接矩阵（相邻为 True 表示判定通过）。
    - 以单元组为整体，按文件名排序后贪心入组：新成员必须与组内每个
      已有成员都满足邻接条件（全连接子图），保证组内任意两张图片都
      被判定为相似。
    - 最终输出分组结果、推荐保留项（按像素数最多选择）以及未归组的
      单张图片，并生成 JSON 报告供人工核对。

输出：
    生成 duplicate_groups.json 报告，包含分组信息、每对的相似度证据、
    推荐保留项等，供人工审核后决定是否删除。
"""
import argparse
import json
from itertools import combinations, islice

import numpy as np
from PIL import Image, ImageOps

from incremental_search import LIBRARY, INDEX_FILE, EXTENSIONS, fingerprint, read_index


def inspect_image(path):
    """
    提取图片的元信息和感知哈希（dHash）。

    功能：
        读取图片文件，自动根据 EXIF 方向信息旋转，然后计算：
        - 图片的原始宽度和高度（像素）
        - 64 位差分感知哈希（dHash），用于快速判断视觉相似性

    参数：
        path: 图片文件的路径（Path 对象或字符串）

    返回：
        dict: 包含以下键的字典：
            - "width" (int): 图片宽度（像素）
            - "height" (int): 图片高度（像素）
            - "dhash" (int): 64 位差分感知哈希值，范围 0 ~ 2^64-1

    算法说明：
        1. 将图片转为灰度图并缩放到 9x8 像素（72 个灰度值）
        2. 比较每行相邻像素的亮度：若右侧像素更亮则为 1，否则为 0
        3. 共 8 行 x 8 个比较 = 64 位，拼接成一个整数
        4. 该哈希对轻微的颜色变化、压缩质量差异具有鲁棒性
    """
    with Image.open(path) as source:
        # 根据 EXIF Orientation 标签自动旋转图片，保证方向正确
        image = ImageOps.exif_transpose(source)
        width, height = image.size
        # 转为灰度图（"L"模式）并缩放到 9x8，使用 Lanczos 抗锯齿算法
        gray = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    # 64 位差分感知哈希：比较每行相邻像素的亮度变化方向
    # gray[:, 1:] 是第 1~8 列，gray[:, :-1] 是第 0~7 列
    # 逐元素比较后展平为一维布尔数组，True 表示右侧更亮（位=1）
    bits = (gray[:, 1:] > gray[:, :-1]).ravel()
    # 将 64 个布尔位拼接成一个 64 位整数
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return {"width": width, "height": height, "dhash": value}


def classify_pair(a, b, info, vectors, similarity, max_distance):
    """
    对一对图片进行分类，判定它们的相似程度类别。

    功能：
        综合 SHA-256 精确哈希、CLIP 余弦相似度和 dHash 汉明距离三个指标，
        将图片对分为四个类别：exact（完全重复）、near（近重复）、
        similar（语义相似但结构差异较大）、other（不相似）。

    参数：
        a (str): 第一张图片的文件名
        b (str): 第二张图片的文件名
        info (dict): 每张图片的元信息字典，键为文件名，值包含：
            - "sha256" (str): 文件内容的 SHA-256 摘要
            - "dhash" (int): 64 位差分感知哈希值
            - "width" (int): 图片宽度
            - "height" (int): 图片高度
        vectors (dict): 每张图片的 CLIP 向量，键为文件名，值为 512 维 numpy 数组
        similarity (float): 余弦相似度阈值（默认 0.94），高于此值认为向量相似
        max_distance (int): dHash 汉明距离阈值（默认 6），低于此值认为结构相似

    返回：
        dict: 包含以下键的字典：
            - "a" (str): 第一张图片的文件名
            - "b" (str): 第二张图片的文件名
            - "category" (str): 分类结果，取值为 "exact"/"near"/"similar"/"other"
            - "cosine" (float): 两张图片 CLIP 向量的余弦相似度，范围 [-1, 1]
            - "dhash_distance" (int): 两张图片 dHash 的汉明距离，范围 0~64

    分类规则（按优先级从高到低）：
        1. exact: SHA-256 完全相同，文件字节级一致，不可拆分
        2. near: CLIP 余弦相似度 >= 阈值 且 dHash 汉明距离 <= 阈值，
                说明语义和视觉结构都高度相似
        3. similar: 仅 CLIP 余弦相似度 >= 阈值，语义相似但结构差异较大
        4. other: 不满足以上任何条件

    注意事项：
        - np.clip 用于将点积结果限制在 [-1, 1] 范围内，防止浮点误差
          导致余弦值略微超出理论范围
        - 异或运算 (^) 后统计 1 的位数即为汉明距离，位差异越大说明
          两张图片的视觉结构差异越大
    """
    # 计算 CLIP 向量的余弦相似度（向量已归一化，点积即余弦）
    # np.clip 防止浮点精度问题导致结果超出 [-1, 1] 理论范围
    score = float(np.clip(vectors[a] @ vectors[b], -1, 1))
    # 计算 dHash 的汉明距离：异或后统计不同位的数量
    distance = (info[a]["dhash"] ^ info[b]["dhash"]).bit_count()
    # 按优先级判定分类
    if info[a]["sha256"] == info[b]["sha256"]:
        # SHA-256 完全一致，文件级重复，最严格的类别
        category = "exact"
    elif score >= similarity and distance <= max_distance:
        # 向量相似度和结构相似度都达标，判定为近重复
        category = "near"
    elif score >= similarity:
        # 仅向量相似度达标，语义相似但视觉结构差异较大
        category = "similar"
    else:
        # 不满足任何相似条件
        category = "other"
    return {"a": a, "b": b, "category": category,
            "cosine": score, "dhash_distance": distance}


def complete_groups(names, matches):
    """
    使用贪心算法将元素分组，要求组内任意两个元素都满足匹配条件。

    功能：
        将给定的元素（文件名或单元组）按稳定排序后，依次尝试加入已有的组。
        新成员必须与组内每个已有成员都满足匹配条件（全连接子图），
        否则创建新组。这是一种保守的分组策略，保证组内质量。

    参数：
        names: 可迭代对象，包含待分组的元素（文件名或单元组代表元）
        matches: 可调用对象，签名为 matches(a, b) -> bool，
                返回两个元素是否满足邻接/匹配条件

    返回：
        list[list]: 分组结果，每个子列表包含一组相互匹配的元素。
                   元素的原始顺序在排序后确定，组按首次创建顺序排列。

    算法说明：
        1. 将所有元素按字典序排序，保证结果稳定可复现
        2. 遍历每个元素，尝试将其加入第一个满足条件的已有组
        3. "满足条件"指：该元素与组内每个成员都通过 matches 判定
        4. 若没有已有组满足条件，则创建新组
        5. 使用 Python for-else 语法：若内层循环正常结束（未 break），
           则执行 else 分支创建新组

    注意事项：
        - 这是贪心算法，结果依赖于排序顺序，不保证找到最少数量的组
        - 但保证了组内的全连接性质：组内任意两个元素都匹配
        - 匹配函数应满足对称性，即 matches(a, b) == matches(b, a)
    """
    groups = []
    # 按字典序排序，保证分组结果的稳定性和可复现性
    for name in sorted(names):
        # 尝试将当前元素加入第一个满足条件的已有组
        for group in groups:
            # 检查当前元素是否与组内每个成员都匹配（全连接条件）
            if all(matches(name, member) for member in group):
                group.append(name)
                break
        else:
            # for 循环正常结束（未触发 break），说明没有合适的已有组，创建新组
            groups.append([name])
    return groups


def generate_groups(library=LIBRARY, index_file=INDEX_FILE, similarity=0.94, max_distance=6, write_report=True):
    """
    主工作流：检测图片库中的重复和近似重复图片，生成分组报告。

    功能：
        完整的重复图片检测流程，包括：
        1. 验证参数和素材一致性
        2. 提取每张图片的元信息和感知哈希
        3. 按 SHA-256 合并完全重复的文件
        4. 分块计算 CLIP 向量相似度，构建邻接矩阵
        5. 使用贪心算法将图片分组
        6. 生成 JSON 报告和可选的控制台输出

    参数：
        library (Path): 图片库目录路径，默认为 LIBRARY 常量
        index_file (Path): 索引文件路径，包含 CLIP 向量和 SHA-256 记录，
                          默认为 INDEX_FILE 常量
        similarity (float): CLIP 余弦相似度阈值，范围 [-1, 1]，默认 0.94。
                           高于此值认为向量相似
        max_distance (int): dHash 汉明距离阈值，范围 [0, 64]，默认 6。
                           低于此值认为结构相似
        write_report (bool): 是否写入 JSON 报告文件，默认 True。
                            若为 False，仅返回报告字典不写文件

    返回：
        dict: 分组报告，包含以下键：
            - "thresholds" (dict): 使用的阈值配置
            - "notice" (str): 提示信息
            - "strategy" (str): 分组策略说明
            - "images" (dict): 每张图片的元信息
            - "groups" (list): 分组列表，每组包含成员、推荐保留项、证据对等
            - "singletons" (list): 未归组的单张图片列表
            - "similar_pairs_not_merged" (list): 高相似度但未合并的图片对样本
            - "similar_pair_count" (int): 高相似度但未合并的图片对总数

    异常：
        ValueError: 参数超出有效范围
        RuntimeError: 素材目录不存在、索引文件缺失、或素材与索引不一致

    工作流程：
        1. 参数验证：检查阈值是否在合理范围内
        2. 一致性检查：确保磁盘上的图片文件与索引记录完全匹配
        3. 元信息提取：为每张图片计算 dHash 和尺寸信息
        4. 完全重复合并：按 SHA-256 将相同文件打包为单元组
        5. 相似度计算：分块计算 CLIP 向量点积，构建布尔邻接矩阵
        6. 贪心分组：以单元组为整体，按文件名排序后贪心入组
        7. 报告生成：输出 JSON 文件和可选的控制台摘要

    注意事项：
        - 不会删除任何图片，仅生成待人工核对的报告
        - 要求素材与索引严格一致，避免对过期向量分组
        - 内存优化：使用布尔邻接矩阵而非字典存储数百万图片对
        - 分块计算：每次处理 128 张图片，避免一次性计算过大的矩阵
    """
    # 参数验证：确保阈值在合理范围内
    if not (-1 <= similarity <= 1) or not (0 <= max_distance <= 64):
        raise ValueError("相似度阈值须在 -1～1，哈希距离须在 0～64。")
    # 检查素材目录和索引文件是否存在
    if not library.is_dir() or not index_file.is_file():
        raise RuntimeError("请先准备素材并执行 python incremental_search.py update")

    # 读取索引：records 是 {文件名: {sha256, ...}} 字典，vectors 是 {文件名: CLIP向量} 字典
    records, vectors = read_index(index_file)
    info = {}
    # 一致性检查：磁盘上的图片文件集合必须与索引记录完全一致
    # 避免对已删除或新增的图片使用过期的向量进行分组
    disk_names = {p.name for p in library.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS}
    if disk_names != set(records):
        raise RuntimeError("素材与成功入库记录不一致。请先 update 并处理失败图片。")

    # 为每张图片提取元信息（尺寸、dHash）并验证 SHA-256 未变化
    for name in sorted(records):
        path = library / name
        digest = fingerprint(path)
        # 验证当前文件的 SHA-256 与索引记录一致，防止图片在索引后被修改
        if digest != records[name]["sha256"]:
            raise RuntimeError(f"图片已变化：{name}。请先运行 update。")
        # 合并 inspect_image 返回的元信息和 SHA-256 摘要
        info[name] = {**inspect_image(path), "sha256": digest}
        # 二次验证：防止在扫描过程中图片被修改（极端情况）
        if fingerprint(path) != digest:
            raise RuntimeError(f"图片在扫描时变化：{name}，请重试。")

    # 第一步：按 SHA-256 将完全相同的文件合并为"单元组"（exact buckets）
    # 单元组内的文件字节级一致，不可拆分，作为后续分组的基本单位
    exact_buckets = {}
    for name in sorted(info):
        exact_buckets.setdefault(info[name]["sha256"], []).append(name)
    # units 字典：以单元组的第一个成员为代表元，值为该组所有成员列表
    units = {members[0]: members for members in exact_buckets.values()}

    # 建立文件名到索引位置的映射，用于快速访问邻接矩阵
    names = sorted(info)
    positions = {name: i for i, name in enumerate(names)}

    # 第二步：构建布尔邻接矩阵，记录哪些图片对满足相似条件
    # 使用 numpy 布尔数组而非 Python 字典，节省内存（避免存储数百万图片对）
    adjacent = np.eye(len(names), dtype=bool)  # 初始化对角线为 True（自己与自己相邻）
    # 将所有 CLIP 向量堆叠为矩阵，便于分块计算点积
    matrix = np.stack([vectors[n] for n in names]) if names else np.empty((0, 512), dtype=np.float32)

    similar_pairs, similar_count = [], 0
    # 分块计算余弦相似度：每次处理 128 张图片，避免一次性计算过大的矩阵
    for start in range(0, len(names), 128):
        # 计算当前块（128 张）与所有图片的点积（即余弦相似度，因为向量已归一化）
        scores = matrix[start:start + 128] @ matrix.T
        for offset, row in enumerate(scores):
            i = start + offset
            # 只检查上三角部分（j > i），避免重复计算和自比较
            # np.flatnonzero 返回满足条件（相似度 >= 阈值）的索引
            for j in np.flatnonzero(row[i + 1:] >= similarity) + i + 1:
                a, b = names[i], names[j]
                # 计算 dHash 汉明距离
                distance = (info[a]["dhash"] ^ info[b]["dhash"]).bit_count()
                # 判定是否满足邻接条件：SHA-256 相同 或 汉明距离在阈值内
                if info[a]["sha256"] == info[b]["sha256"] or distance <= max_distance:
                    # 满足条件，标记为相邻（双向）
                    adjacent[i, j] = adjacent[j, i] = True
                else:
                    # 向量相似度高但 dHash 距离大，记录为"相似但未合并"的对
                    # 这类对可能帮助排查误判，但不足以入组
                    similar_count += 1
                    if len(similar_pairs) < 200:
                        similar_pairs.append(classify_pair(a, b, info, vectors, similarity, max_distance))

    # 完全重复组内部强制标记为相邻，不依赖向量分数
    # 原因：浮点精度可能导致相同文件的向量点积略低于 1.0，但仍应视为完全重复
    for members in units.values():
        ids = [positions[name] for name in members]
        # np.ix_ 用于选择子矩阵，将单元组内所有图片对标记为相邻
        adjacent[np.ix_(ids, ids)] = True

    def pair(a, b):
        """辅助函数：对两张图片调用 classify_pair 进行分类"""
        return classify_pair(a, b, info, vectors, similarity, max_distance)

    # 定义单元组之间的匹配函数：两个单元组匹配当且仅当它们之间的所有图片对都相邻
    # 这保证了最终分组的全连接性质：组内任意两张图片都满足相似条件
    def matches(a, b):
        return all(adjacent[positions[x], positions[y]]
                   for x in units[a] for y in units[b])

    # 第三步：使用贪心算法将单元组分组
    # complete_groups 会按字典序遍历单元组代表元，尝试加入已有组
    grouped_units = complete_groups(units, matches)

    # 第四步：展开单元组，生成最终的分组结果
    groups, singles = [], []
    for unit_group in grouped_units:
        # 将单元组展开为所有成员文件的平铺列表
        members = sorted(name for unit in unit_group for name in units[unit])
        if len(members) == 1:
            # 单张图片，未归入任何重复组
            singles.extend(members)
            continue
        # 收集组内图片对的分类证据，最多 200 对（避免输出过大）
        evidence = [pair(a, b) for a, b in islice(combinations(members, 2), 200)]
        # 推荐保留项：按像素数量最多选择，并列时按文件名字典序
        # 注意：仅按像素数量推荐，不声称分辨率最高就一定质量最好
        keeper = min(members, key=lambda n: (-info[n]["width"] * info[n]["height"], n))
        groups.append({"id": len(groups) + 1,
                       # 若单元组数量为 1，说明是完全重复组；否则为近重复组
                       "category": "exact" if len(unit_group) == 1 else "near",
                       "members": members, "suggested_keep": keeper,
                       "keep_reason": "像素数量最多；并列按文件名排序，需人工确认",
                       "pairs": evidence,
                       # 组内图片对的总数（组合数 C(n,2)）
                       "pair_count": len(members) * (len(members) - 1) // 2})

    # 构建报告字典，包含所有分组信息和辅助数据
    report = {
        "thresholds": {"cosine": similarity, "dhash_distance": max_distance},
        "notice": "实验阈值，结果待人工核对；不执行删除。相似标签仅指向量达到阈值而哈希未通过。",
        "strategy": "先合并完全重复文件，再按文件名顺序入组，组内所有图片对须通过判定",
        "images": info, "groups": groups, "singletons": singles,
        # 高相似度但未通过近重复规则的图片对，帮助排查误判
        "similar_pairs_not_merged": similar_pairs,
        "similar_pair_count": similar_count,
    }
    output = index_file.parent / "duplicate_groups.json"
    if write_report:
        # 写入 JSON 报告文件，使用 UTF-8 编码和 2 空格缩进
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not write_report:
        return report

    # 控制台输出摘要信息
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
    # 命令行入口：允许用户通过参数调整相似度阈值和哈希距离阈值
    parser = argparse.ArgumentParser(description=__doc__)
    # --similarity: CLIP 余弦相似度阈值，越高越严格，默认 0.94
    parser.add_argument("--similarity", type=float, default=0.94)
    # --hash-distance: dHash 汉明距离阈值，越低越严格，默认 6
    parser.add_argument("--hash-distance", type=int, default=6)
    args = parser.parse_args()
    # 使用命令行参数调用主工作流
    generate_groups(similarity=args.similarity, max_distance=args.hash_distance)
