"""Persistent vector retrieval with stable IDs, FAISS and NumPy fallback."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


MANIFEST_VERSION = 2


def _faiss():
    try:
        import faiss
        return faiss
    except ImportError:
        return None


def backend_name():
    return "faiss-persistent-idmap2" if _faiss() is not None else "numpy-dot-product"


def _matrix(names, vectors):
    matrix = np.ascontiguousarray(np.stack([vectors[name] for name in names]), dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix /= np.maximum(norms, 1e-12)
    return matrix


def _query(query, dimension):
    query = np.asarray(query, dtype=np.float32).reshape(1, -1)
    if query.shape[1] != dimension or not np.isfinite(query).all():
        raise ValueError("查询向量维度无效")
    query /= max(float(np.linalg.norm(query)), 1e-12)
    return np.ascontiguousarray(query)


def rank_vectors(records, vectors, query, limit=None):
    """In-memory ranking fallback used when no persistent sidecar is available."""
    names = sorted(name for name in records if name in vectors)
    if not names:
        return [], backend_name()
    matrix = _matrix(names, vectors)
    query = _query(query, matrix.shape[1])
    count = len(names) if limit is None else max(1, min(int(limit), len(names)))
    faiss = _faiss()
    if faiss is not None:
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        scores, positions = index.search(query, count)
        rows = [(names[int(position)], float(score))
                for score, position in zip(scores[0], positions[0]) if position >= 0]
        return rows, "faiss-transient-flat-ip"
    scores = matrix @ query[0]
    order = np.argsort(-scores, kind="stable")[:count]
    return [(names[int(i)], float(scores[int(i)])) for i in order], "numpy-dot-product"


def threshold_pairs(names, vectors, threshold, block_size=128):
    """Return deterministic candidate pairs without constructing N x N state."""
    names = list(names)
    if not -1 <= float(threshold) <= 1:
        raise ValueError("相似度阈值须在 -1～1")
    if len(names) < 2:
        return [], "faiss-range-search" if _faiss() is not None else "numpy-block-scan"
    matrix = _matrix(names, vectors)
    candidates = []
    faiss = _faiss()
    if faiss is not None:
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        limits, scores, positions = index.range_search(matrix, float(threshold) - 1e-7)
        for left in range(len(names)):
            for offset in range(int(limits[left]), int(limits[left + 1])):
                right = int(positions[offset])
                score = float(scores[offset])
                if right > left and score >= threshold:
                    candidates.append((left, right, score))
        return sorted(candidates), "faiss-range-search"
    block_size = max(1, int(block_size))
    for start in range(0, len(names), block_size):
        scores = matrix[start:start + block_size] @ matrix.T
        for offset, row in enumerate(scores):
            left = start + offset
            for right in np.flatnonzero(row[left + 1:] >= threshold) + left + 1:
                candidates.append((left, int(right), float(row[right])))
    return candidates, "numpy-block-scan"


def _paths(index_file):
    index_file = Path(index_file)
    return index_file.with_suffix(".faiss"), index_file.with_suffix(".faiss.json")


def _stable_id(name, occupied):
    value = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big") & ((1 << 63) - 1)
    value = value or 1
    while value in occupied:
        value = 1 if value == (1 << 63) - 1 else value + 1
    return value


def _read_manifest(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != MANIFEST_VERSION or not isinstance(data.get("items"), dict):
            return None
        return data
    except (OSError, ValueError, TypeError):
        return None


def _load_faiss(path, faiss):
    try:
        return faiss.deserialize_index(np.frombuffer(path.read_bytes(), dtype=np.uint8))
    except (OSError, RuntimeError, ValueError):
        return None


def _write_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=str(path.parent), suffix=path.suffix)
    os.close(handle)
    temporary = Path(name)
    try:
        temporary.write_bytes(data)
        os.replace(str(temporary), str(path))
    finally:
        temporary.unlink(missing_ok=True)


def persist_vector_index(index_file, records, vectors):
    """Incrementally synchronize a persistent ``IndexIDMap2`` sidecar."""
    faiss = _faiss()
    if faiss is None:
        return "numpy-dot-product"
    names = sorted(name for name in records if name in vectors)
    dimension = int(np.asarray(vectors[names[0]]).size) if names else 512
    faiss_path, manifest_path = _paths(index_file)
    old_manifest = _read_manifest(manifest_path)
    index = _load_faiss(faiss_path, faiss) if old_manifest else None
    if index is None or int(getattr(index, "d", -1)) != dimension:
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
        old_manifest = {"items": {}}
    old_items = old_manifest.get("items", {})
    occupied = {int(item["id"]) for item in old_items.values() if "id" in item}
    desired = {}
    for name in names:
        previous = old_items.get(name)
        vector_id = int(previous["id"]) if previous and "id" in previous else _stable_id(name, occupied)
        occupied.add(vector_id)
        desired[name] = {"id": vector_id, "sha256": records[name].get("sha256")}
    remove_ids = [int(item["id"]) for name, item in old_items.items()
                  if name not in desired or item.get("sha256") != desired[name].get("sha256")]
    if remove_ids:
        index.remove_ids(np.asarray(remove_ids, dtype=np.int64))
    changed = [name for name in names
               if name not in old_items or old_items[name].get("sha256") != desired[name].get("sha256")]
    if changed:
        index.add_with_ids(_matrix(changed, vectors),
                           np.asarray([desired[name]["id"] for name in changed], dtype=np.int64))
    if int(index.ntotal) != len(names):
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
        if names:
            index.add_with_ids(_matrix(names, vectors),
                               np.asarray([desired[name]["id"] for name in names], dtype=np.int64))
    _write_atomic(faiss_path, faiss.serialize_index(index).tobytes())
    _write_atomic(manifest_path, json.dumps({
        "version": MANIFEST_VERSION, "dimension": dimension, "items": desired,
    }, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8"))
    return "faiss-persistent-idmap2"


def rank_persistent(index_file, records, vectors, query, limit=None):
    """Search the persisted FAISS index, rebuilding only when it is stale."""
    faiss = _faiss()
    if faiss is None:
        return rank_vectors(records, vectors, query, limit)
    names = sorted(name for name in records if name in vectors)
    if not names:
        return [], backend_name()
    faiss_path, manifest_path = _paths(index_file)
    manifest = _read_manifest(manifest_path)
    expected = {name: records[name].get("sha256") for name in names}
    current = ({name: item.get("sha256") for name, item in manifest["items"].items()} if manifest else {})
    index = _load_faiss(faiss_path, faiss) if manifest and current == expected else None
    if index is None or int(index.ntotal) != len(names):
        persist_vector_index(index_file, records, vectors)
        manifest = _read_manifest(manifest_path)
        index = _load_faiss(faiss_path, faiss)
    if index is None or manifest is None:
        return rank_vectors(records, vectors, query, limit)
    count = len(names) if limit is None else max(1, min(int(limit), len(names)))
    scores, ids = index.search(_query(query, int(index.d)), count)
    id_to_name = {int(item["id"]): name for name, item in manifest["items"].items()}
    rows = [(id_to_name[int(vector_id)], float(score))
            for score, vector_id in zip(scores[0], ids[0]) if int(vector_id) in id_to_name]
    return rows, "faiss-persistent-idmap2"
