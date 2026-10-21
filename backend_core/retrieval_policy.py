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
        category, reason = "相似", "CLIP 视觉语义相近，但未达到近重复的双重条件"
    else:
        category, reason = "相关候选", "按 CLIP 余弦相似度排序，建议人工确认"
    return {"category": category, "reason": reason, "dhash_distance": distance,
            "phash_distance": perceptual_distance,
            "color_distance": round(chroma_distance, 6)}


def explain_text_result(score, text):
    level = "高相关" if score >= 0.30 else "中等相关" if score >= 0.20 else "弱相关"
    return {"category": level,
            "reason": f"文本“{text[:40]}”与图片在 OpenCLIP 共享向量空间中的余弦相似度为 {score:.4f}"}
