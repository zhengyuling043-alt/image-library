"""Measure real cached OpenCLIP single-image versus batch inference."""

import argparse
import json
import platform
import time
from pathlib import Path

import psutil

from backend_core.model_service import encode_image_path, encode_image_paths, load_openclip_model, select_device
from indexed_search import MODEL_CACHE_DIR, MODEL_NAME, WEIGHTS


ROOT = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    paths = sorted((ROOT / "data" / "library").glob("*"))[:max(1, args.images)]
    if not paths:
        raise RuntimeError("素材库为空")
    process = psutil.Process()
    memory_before = process.memory_info().rss
    started = time.perf_counter()
    model, preprocess = load_openclip_model(MODEL_NAME, WEIGHTS, MODEL_CACHE_DIR)
    load_seconds = time.perf_counter() - started
    encode_image_path(paths[0], model, preprocess)  # warm-up
    started = time.perf_counter()
    for path in paths:
        encode_image_path(path, model, preprocess)
    single_seconds = time.perf_counter() - started
    started = time.perf_counter()
    vectors, failures = encode_image_paths(paths, model, preprocess, args.batch_size)
    batch_seconds = time.perf_counter() - started
    payload = {
        "scope": "缓存权重后的真实OpenCLIP CPU/GPU推理；包含图片解码与预处理，不包含下载。",
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "device": select_device()},
        "model": {"name": MODEL_NAME, "weights": WEIGHTS, "images": len(paths),
                  "batch_size": args.batch_size},
        "results": {"load_seconds": round(load_seconds, 3),
                    "single_total_seconds": round(single_seconds, 3),
                    "single_images_per_second": round(len(paths) / single_seconds, 3),
                    "batch_total_seconds": round(batch_seconds, 3),
                    "batch_images_per_second": round(len(vectors) / batch_seconds, 3),
                    "speedup": round(single_seconds / batch_seconds, 3),
                    "failures": {str(path): reason for path, reason in failures.items()},
                    "rss_mb": round((process.memory_info().rss - memory_before) / 1024 / 1024, 2)},
    }
    (ROOT / "MODEL_BATCH_BENCHMARK_RESULTS.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result = payload["results"]
    lines = ["# OpenCLIP真实批处理测试", "", payload["scope"], "",
             f"- 模型：`{MODEL_NAME}` / `{WEIGHTS}`；设备：`{payload['runtime']['device']}`；样本：{len(paths)} 张。",
             f"- 模型加载：{result['load_seconds']} s（权重已缓存于D盘）。",
             f"- 逐张编码：{result['single_total_seconds']} s，{result['single_images_per_second']} 张/s。",
             f"- 批量编码：{result['batch_total_seconds']} s，{result['batch_images_per_second']} 张/s。",
             f"- 本机实测加速比：{result['speedup']}x；RSS变化：{result['rss_mb']} MB。", "",
             "命令：`.venv\\Scripts\\python.exe benchmark_model_batch.py --images 8 --batch-size 8`。",
             "不同CPU/GPU和冷启动状态会改变结果；报告不外推未实测的5000张模型耗时。"]
    (ROOT / "MODEL_BATCH_BENCHMARK_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
