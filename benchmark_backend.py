"""Reproducible vector-search benchmark for 500/2000/5000 item libraries."""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import psutil
import faiss


ROOT = Path(__file__).parent


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))]


def benchmark(size, dimension=512, queries=40, seed=42):
    rng = np.random.default_rng(seed + size)
    matrix = rng.normal(size=(size, dimension)).astype(np.float32)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    query_matrix = rng.normal(size=(queries, dimension)).astype(np.float32)
    query_matrix /= np.maximum(np.linalg.norm(query_matrix, axis=1, keepdims=True), 1e-12)
    process = psutil.Process()
    memory_before = process.memory_info().rss

    start = time.perf_counter()
    index = faiss.IndexFlatIP(dimension)
    index.add(np.ascontiguousarray(matrix))
    build_ms = (time.perf_counter() - start) * 1000

    faiss_times, numpy_times = [], []
    for query in query_matrix:
        start = time.perf_counter()
        index.search(query.reshape(1, -1), 20)
        faiss_times.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        scores = matrix @ query
        np.argpartition(scores, -min(20, size))[-min(20, size):]
        numpy_times.append((time.perf_counter() - start) * 1000)

    memory_after = process.memory_info().rss
    return {
        "images": size, "dimension": dimension, "queries": queries,
        "faiss_build_ms": round(build_ms, 3),
        "faiss_p50_ms": round(statistics.median(faiss_times), 4),
        "faiss_p95_ms": round(percentile(faiss_times, .95), 4),
        "numpy_p50_ms": round(statistics.median(numpy_times), 4),
        "numpy_p95_ms": round(percentile(numpy_times, .95), 4),
        "rss_delta_mb": round(max(0, memory_after - memory_before) / 1024 / 1024, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[500, 2000, 5000])
    parser.add_argument("--queries", type=int, default=40)
    args = parser.parse_args()
    results = [benchmark(size, queries=args.queries) for size in args.sizes]
    payload = {
        "scope": "仅测试已编码512维向量的检索；不包含OpenCLIP图片编码和磁盘扫描时间。",
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "numpy": np.__version__},
        "results": results,
    }
    (ROOT / "BENCHMARK_RESULTS.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = ["# 后端向量检索性能测试", "", payload["scope"], "",
             f"运行环境：Python {payload['runtime']['python']}；NumPy {payload['runtime']['numpy']}。", "",
             "| 图片数 | FAISS构建(ms) | FAISS P50(ms) | FAISS P95(ms) | NumPy P50(ms) | NumPy P95(ms) | RSS增量(MB) |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in results:
        lines.append("| {images} | {faiss_build_ms} | {faiss_p50_ms} | {faiss_p95_ms} | "
                     "{numpy_p50_ms} | {numpy_p95_ms} | {rss_delta_mb} |".format(**row))
    lines += ["", "命令：`.venv\\Scripts\\python.exe benchmark_backend.py`。",
              "随机种子固定；同一机器可复现，但不同硬件结果会变化。"]
    (ROOT / "BENCHMARK_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
