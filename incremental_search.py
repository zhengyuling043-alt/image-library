"""单进程命令行增量索引：文件哈希检测变化，单文件原子保存。"""
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
LIBRARY = ROOT / "data" / "library"
INDEX_FILE = ROOT / "data" / "index" / "incremental.npz"
MODEL = "ViT-B-32-quickgelu"
WEIGHTS = "openai"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Encoder:
    def __init__(self):
        self.model = None

    def __call__(self, path):
        # 无变化时不加载模型，也不访问模型下载平台。
        from indexed_search import load_model, encode_image
        if self.model is None:
            self.model, self.preprocess = load_model()
        return encode_image(path, self.model, self.preprocess)


def read_index(path):
    if not path.exists():
        return {}, {}
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        vectors = archive["vectors"]
    if (metadata.get("model"), metadata.get("weights")) != (MODEL, WEIGHTS):
        raise RuntimeError("模型配置不同，请使用 --rebuild 重建增量索引。")
    records = metadata["records"]
    names = list(records)
    if vectors.ndim != 2 or len(names) != len(vectors):
        raise RuntimeError("增量索引不一致，请使用 --rebuild 重建。")
    if not np.isfinite(vectors).all():
        raise RuntimeError("索引含无效向量，请重新建立。")
    return records, dict(zip(names, vectors))


def save_index(path, records, vectors):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.stack([vectors[name] for name in records]) if records else np.empty((0, 512), dtype=np.float32)
    metadata = json.dumps({"model": MODEL, "weights": WEIGHTS, "records": records}, ensure_ascii=False)
    # 同一文件中保存向量和元数据，替换前旧索引仍可用。
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez(stream, vectors=array, metadata=np.array(metadata))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def update_index(library=LIBRARY, index_file=INDEX_FILE, encoder=None, rebuild=False):
    if not library.is_dir():
        raise RuntimeError(f"素材文件夹不存在：{library}，请检查路径。")
    old_records, old_vectors = ({}, {}) if rebuild else read_index(index_file)
    files = sorted(p for p in library.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    current_names = {p.name for p in files}
    counts = dict(added=0, modified=0, reused=0, deleted=len(set(old_records) - current_names), failed=0)
    records, vectors, failures = {}, {}, {}
    encoder = encoder or Encoder()
    for path in files:
        name = path.name
        try:
            digest = fingerprint(path)
            if old_records.get(name, {}).get("sha256") == digest:
                vector = old_vectors[name]
                counts["reused"] += 1
            else:
                print(f"计算图片向量：{name}")
                vector = np.asarray(encoder(path), dtype=np.float32)
                if vector.shape != (512,) or not np.isfinite(vector).all():
                    raise ValueError("模型输出不是有效的 512 维向量")
                if fingerprint(path) != digest:
                    raise ValueError("图片在处理期间发生变化，请稍后重试")
                counts["modified" if name in old_records else "added"] += 1
            records[name] = {"sha256": digest, "status": "ready"}
            vectors[name] = vector
        except (OSError, ValueError) as error:
            counts["failed"] += 1
            failures[name] = str(error)
            print(f"入库失败：{name}：{error}")
            # 不继续搜索已修改但读取失败的图片；下次 update 自动重试。
    save_index(index_file, records, vectors)
    report = {"counts": counts, "ready": len(records), "failures": failures}
    index_file.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n新增 {counts['added']}，修改 {counts['modified']}，复用 {counts['reused']}，删除 {counts['deleted']}，失败 {counts['failed']}")
    print(f"当前可搜索图片：{len(records)} 张")
    return report


def search(query, library=LIBRARY, index_file=INDEX_FILE):
    if not index_file.is_file():
        raise RuntimeError("请先运行 python incremental_search.py update")
    if not query.is_file():
        raise FileNotFoundError(f"查询图不存在：{query}")
    records, vectors = read_index(index_file)
    if not records:
        print("索引为空，请加入图片后运行 update。")
        return
    print(f"读取已保存的 {len(records)} 张素材向量，只编码查询图。")
    query_vector = Encoder()(query)
    results = []
    stale = 0
    for name, record in records.items():
        try:
            if fingerprint(library / name) != record["sha256"]:
                stale += 1
                continue
        except OSError:
            stale += 1
            continue
        results.append((float(vectors[name] @ query_vector), name))
    if stale:
        print(f"已跳过 {stale} 张已删除或变化的素材，请运行 update 同步。")
    print(f"\n查询图片：{query.name}；最相似的前 5 张：")
    for rank, (score, name) in enumerate(sorted(results, key=lambda x: (-x[0], x[1]))[:5], 1):
        print(f"{rank}. 相似度 {score:.4f} | {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["update", "search"])
    parser.add_argument("--query", default="data/test.jpg")
    parser.add_argument("--rebuild", action="store_true", help="放弃旧增量缓存并重新编码")
    args = parser.parse_args()
    if args.action == "update":
        update_index(rebuild=args.rebuild)
    else:
        query = Path(args.query)
        search(query if query.is_absolute() else ROOT / query)
