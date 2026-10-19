"""Vector retrieval backend with FAISS acceleration and NumPy fallback."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np


def _faiss():
    try:
        import faiss
        return faiss
    except ImportError:
        return None


def backend_name():
    return "faiss-index-flat-ip" if _faiss() is not None else "numpy-dot-product"


def _matrix(names, vectors):
    matrix = np.ascontiguousarray(np.stack([vectors[name] for name in names]), dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix /= np.maximum(norms, 1e-12)
    return matrix


def rank_vectors(records, vectors, query, limit=None):
    """Rank indexed vectors by cosine similarity."""

    names = sorted(name for name in records if name in vectors)
    if not names:
        return [], backend_name()
    matrix = _matrix(names, vectors)
    query = np.asarray(query, dtype=np.float32).reshape(1, -1)
    if query.shape[1] != matrix.shape[1] or not np.isfinite(query).all():
        raise ValueError("查询向量维度无效")
    query /= max(float(np.linalg.norm(query)), 1e-12)
    count = len(names) if limit is None else max(1, min(int(limit), len(names)))
    faiss = _faiss()
    if faiss is not None:
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        scores, positions = index.search(np.ascontiguousarray(query), count)
        rows = [(names[int(position)], float(score))
                for score, position in zip(scores[0], positions[0]) if position >= 0]
        return rows, "faiss-index-flat-ip"
    scores = matrix @ query[0]
    order = np.argsort(-scores, kind="stable")[:count]
    return [(names[int(i)], float(scores[int(i)])) for i in order], "numpy-dot-product"


def persist_vector_index(index_file, records, vectors):
    """Persist a FAISS sidecar when FAISS is installed."""

    faiss = _faiss()
    if faiss is None:
        return "numpy-dot-product"
    names = sorted(name for name in records if name in vectors)
    matrix = _matrix(names, vectors) if names else np.empty((0, 512), dtype=np.float32)
    index = faiss.IndexFlatIP(matrix.shape[1] if matrix.ndim == 2 and matrix.shape[1] else 512)
    if len(matrix):
        index.add(matrix)
    index_file = Path(index_file)
    faiss_path = index_file.with_suffix(".faiss")
    names_path = index_file.with_suffix(".faiss.json")
    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        handle, name = tempfile.mkstemp(dir=str(faiss_path.parent), suffix=".faiss")
        os.close(handle)
        temporary = Path(name)
        # FAISS 的 Windows FileIOWriter 不能可靠处理非 ASCII 路径。
        # 先序列化到内存，再由 Python 写盘即可支持中文用户名与目录。
        serialized = faiss.serialize_index(index)
        temporary.write_bytes(serialized.tobytes())
        os.replace(str(temporary), str(faiss_path))
        names_path.write_text(json.dumps({"names": names}, ensure_ascii=False), encoding="utf-8")
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return "faiss-index-flat-ip"
