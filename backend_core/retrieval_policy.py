"""Auditable retrieval thresholds and result explanations."""

from group_duplicates import (
    color_distance, hash_distance, near_duplicate_decision, phash_distance,
)


PRESETS = {
    "strict": {"label": "严格去重", "cosine": 0.94, "dhash_distance": 6,
               "description": "优先准确率，适合自动清理前复核"},
    "balanced": {"label": "校准推荐", "cosine": 0.86, "dhash_distance": 8,
                 "description": "由校准集选出的精确率与召回率平衡点"},
    "broad": {"label": "宽松发现", "cosine": 0.78, "dhash_distance": 28,
              "description": "优先召回率，适合人工浏览候选"},
}


def preset(name="balanced"):
    if name not in PRESETS:
        raise ValueError("未知阈值预设")
    return dict(PRESETS[name])


def explain_image_result(score, query_sha, query_info, candidate_sha, candidate_info):
    distance = hash_distance(query_info, candidate_info)
    perceptual_distance = phash_distance(query_info, candidate_info)
    chroma_distance = color_distance(query_info, candidate_info)
    if query_sha == candidate_sha:
        category, reason = "重复", "文件内容 SHA-256 完全一致"
    elif near_duplicate_decision(score, distance, perceptual_distance, chroma_distance,
                                 PRESETS["balanced"]["cosine"],
                                 PRESETS["balanced"]["dhash_distance"]):
        category, reason = "近重复", f"CLIP、dHash {distance}、pHash {perceptual_distance} 与RGB色度 {chroma_distance:.3f} 共同通过"
    elif score >= PRESETS["broad"]["cosine"]:
        category, reason = "相似", "CLIP 语义相近但哈希结构校验未通过，属于不同图片，不会合并删除"
    else:
        category, reason = "相关候选", "按 CLIP 余弦相似度排序，建议人工确认"
    return {"category": category, "reason": reason, "dhash_distance": distance,
            "phash_distance": perceptual_distance,
            "color_distance": round(chroma_distance, 6),
            "interpretation": _interpret_image_score(score, category)}


def _interpret_image_score(score, category):
    """为不熟悉 CLIP/dHash 的用户给出分数通俗解读。

    以图搜图的分数含义（图与图之间）：
      - 0.94 以上：高度疑似同一张图或近重复变体（裁剪/缩放/调色/水印等）；
      - 0.80 ～ 0.94：视觉语义相近，通常是“同一类内容”（例如两张不同的狗照片），
        不代表是同一张图，是否同源要结合 dHash 距离判断；
      - 0.60 ～ 0.80：内容上有一定相关性（同场景/同物体大类）；
      - 0.60 以下：相关性较弱，多为背景或构图层面的偶然相似。
    """
    if category in ("重复", "近重复"):
        return "分数很高：通常代表同一张图片，或仅经过裁剪/缩放/调色/加印等修改的版本。"
    if score >= 0.90:
        return "分数较高：极可能是同一张图的修改版本，但仍需哈希距离确认。"
    if score >= 0.80:
        return "中等偏高：说明两张图属于同一类内容（如都是狗/山景），并不代表是同一张图。"
    if score >= 0.60:
        return "中等：仅表示内容或场景有相关性，通常不是同一张图。"
    return "较低：仅存在轻微背景或构图上的相似，一般与查询图关联不大。"


def explain_text_result(score, text):
    level = "高相关" if score >= 0.30 else "中等相关" if score >= 0.20 else "弱相关"
    return {"category": level,
            "reason": f"文本“{text[:40]}”与图片在 OpenCLIP 共享向量空间中的余弦相似度为 {score:.4f}",
            "interpretation": _interpret_text_score(score)}


def _interpret_text_score(score):
    """为不熟悉 CLIP 的用户给出文本搜索分数通俗解读。

    文本→图像的余弦相似度天然比“图→图”低一个量级，
    0.20～0.35 即已属于语义匹配的正常区间，请不要与以图搜图的分数直接对比。
    """
    if score >= 0.30:
        return "高相关：文本与图片语义一致（文本搜索 0.30 以上已是很好的匹配，不宜用图搜图的标准衡量）。"
    if score >= 0.20:
        return "中等相关：文本描述能在图片中找到对应内容（文本搜索的正常区间为 0.20～0.30）。"
    return "弱相关：文本与图片内容关联不大，可尝试更换英文关键词或降低相似度下限。"
