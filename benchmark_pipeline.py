"""Measure the real import/index/search pipeline at 500/2000/5000 items.

The benchmark intentionally uses a deterministic lightweight encoder so the
reported numbers isolate filesystem scan, hashing, checkpoint persistence,
SQLite metadata and persistent FAISS behavior. OpenCLIP inference is measured
separately by ``benchmark_model_batch.py`` and is never disguised as part of
these timings.
"""

import argparse
import json
import platform
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import psutil
from PIL import Image, ImageDraw

from incremental_search import read_index, update_index
from vector_index import rank_persistent


ROOT = Path(__file__).parent


class DeterministicEncoder:
    def encode_many(self, paths, batch_size=None):
        vectors = {}
        for path in paths:
            number = int(path.stem.split("_")[-1])
            vector = np.zeros(512, np.float32)
            vector[number % 512] = 1.0
            vector[(number * 17 + 3) % 512] += 0.5
            vector /= np.linalg.norm(vector)
            vectors[path] = vector
        return vectors, {}


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))]


def make_images(library, size):
    library.mkdir(parents=True)
    for number in range(size):
        image = Image.new("RGB", (48, 48), ((number * 31) % 256, (number * 67) % 256, (number * 97) % 256))
        ImageDraw.Draw(image).line((0, number % 48, 47, (number * 7) % 48), fill="white", width=2)
        image.save(library / f"bench_{number:05d}.png", optimize=False)


def benchmark(size, queries=40):
    runtime = ROOT / ".runtime" / "pipeline-benchmark"
    runtime.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    with tempfile.TemporaryDirectory(dir=runtime) as directory:
        root = Path(directory)
        library, index = root / "library", root / "index" / "library.npz"
        make_images(library, size)
        memory_before = process.memory_info().rss
        start = time.perf_counter()
        first = update_index(library, index, DeterministicEncoder(), checkpoint_every=250)
        import_seconds = time.perf_counter() - start
        memory_after = process.memory_info().rss

        start = time.perf_counter()
        reused = update_index(library, index, DeterministicEncoder(), checkpoint_every=250)
        reuse_seconds = time.perf_counter() - start

        Image.new("RGB", (48, 48), "navy").save(library / f"bench_{size:05d}.png")
        (library / "bench_00000.png").unlink()
        start = time.perf_counter()
        incremental = update_index(library, index, DeterministicEncoder(), checkpoint_every=250)
        incremental_ms = (time.perf_counter() - start) * 1000

        records, vectors = read_index(index)
        query = next(iter(vectors.values()))
        timings = []
        backend = None
        for _ in range(queries):
            start = time.perf_counter()
            _, backend = rank_persistent(index, records, vectors, query, limit=20)
            timings.append((time.perf_counter() - start) * 1000)
        return {
            "images": size,
            "initial_import_seconds": round(import_seconds, 3),
            "throughput_images_per_second": round(size / import_seconds, 1),
            "no_change_rescan_seconds": round(reuse_seconds, 3),
            "incremental_add_delete_ms": round(incremental_ms, 2),
            "search_p50_ms": round(statistics.median(timings), 4),
            "search_p95_ms": round(percentile(timings, .95), 4),
            "rss_delta_mb": round(max(0, memory_after - memory_before) / 1024 / 1024, 2),
            "backend": backend,
            "counts": {"initial": first["counts"], "reuse": reused["counts"],
                       "incremental": incremental["counts"]},
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[500, 2000, 5000])
    parser.add_argument("--queries", type=int, default=40)
    args = parser.parse_args()
    rows = [benchmark(size, args.queries) for size in args.sizes]
    payload = {
        "scope": "真实扫描/哈希/NPZ/SQLite/持久FAISS/检索管线；编码器为确定性替身，OpenCLIP另测。",
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "results": rows,
    }
    (ROOT / "PIPELINE_BENCHMARK_RESULTS.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = ["# 端到端后端管线规模测试", "", payload["scope"], "",
             "| 图片数 | 首次入库(s) | 吞吐(张/s) | 无变化重扫(s) | 增1删1(ms) | 搜索P50(ms) | 搜索P95(ms) | RSS增量(MB) |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| {images} | {initial_import_seconds} | {throughput_images_per_second} | "
                     "{no_change_rescan_seconds} | {incremental_add_delete_ms} | {search_p50_ms} | "
                     "{search_p95_ms} | {rss_delta_mb} |".format(**row))
    lines += ["", "首次入库时间不包含测试图片生成；包含文件扫描、SHA-256、索引原子保存、SQLite同步和FAISS持久化。",
              "编码器是固定512维替身，因此这些数值不能当作OpenCLIP推理耗时。",
              "命令：`.venv\\Scripts\\python.exe benchmark_pipeline.py`。"]
    (ROOT / "PIPELINE_BENCHMARK_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
