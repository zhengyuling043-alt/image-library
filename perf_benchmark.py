"""性能基准测试：按课程要求分 500/2000/5000 张三档素材库规模，验证性能指标。

【测试图片来源】本脚本不依赖任何真实素材：
  图片全部由下方 make_image() 程序合成（随机色块图，每张几毫秒），
  生成在临时目录（tempfile.TemporaryDirectory），测试结束自动删除，
  全程不碰 data/library 中的真实素材库，也不会污染正式索引。
  真实 CLIP 模型从本地权重 data/models/open_clip_pytorch_model.bin 加载。

【测试结果去向】每次运行写入项目根目录 perf_results_<规模>.json（UTF-8），
  例如 py -s perf_benchmark.py 500 → perf_results_500.json。
  各字段含义（对应性能测试 5 项要求）：
  full_build_sec            批量入库总耗时（全量建索引：SHA-256 扫描 + CLIP 编码 + 保存）；
  incremental_add_N_sec     增量更新耗时——新增 10% 图片只编码新图，与全量对比证明增量索引优势；
  search_sec                以图搜图响应时间（查询图编码 + 全库余弦比对 + 排序，要求秒级）；
  grouping_sec              全库重复分组耗时（dHash + 余弦矩阵 + 聚类）；
  mem_*_mb                  内存占用（RSS 实测：模型加载前后、峰值）；
  index_file_mb             向量索引磁盘占用（验证素材量增长对存储的影响）。

【已知问题】内存读数（mem_*_mb）在部分 Windows 环境读不到，结果为 0.0，
  使用前请先验证 rss_mb() 是否返回有效值；详见下方 rss_mb() 注释。

用法：py -s perf_benchmark.py <规模>   （例如 py -s perf_benchmark.py 500）
"""

import sys
import time
import json
import random
import tempfile
import ctypes
from pathlib import Path
from ctypes import wintypes

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from incremental_search import update_index, search  # noqa: E402
from group_duplicates import generate_groups  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402


class _PMC(ctypes.Structure):
    """Windows 进程内存计数器结构体，用于读取内存占用"""
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def rss_mb():
    """读取当前进程物理内存占用（MB）"""
    # 显式声明函数原型：64 位下不设置 argtypes/restype 会导致指针截断、读数恒为 0
    _get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    _get_current_process.argtypes = []
    _get_current_process.restype = wintypes.HANDLE
    _get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    _get_process_memory_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
    _get_process_memory_info.restype = wintypes.BOOL

    counters = _PMC()
    counters.cb = ctypes.sizeof(_PMC)
    handle = _get_current_process()
    if not _get_process_memory_info(handle, ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    return counters.WorkingSetSize / 1048576


def make_image(path: Path, rng: random.Random):
    """快速生成一张随机构图的合成图片（纯色底 + 随机色块），几毫秒一张"""
    w, h = rng.randint(300, 900), rng.randint(300, 900)
    img = Image.new("RGB", (w, h),
                    (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)))
    draw = ImageDraw.Draw(img)
    for _ in range(rng.randint(2, 6)):
        x1, y1 = rng.randint(0, w - 1), rng.randint(0, h - 1)
        x2, y2 = rng.randint(x1, w - 1), rng.randint(y1, h - 1)
        draw.rectangle([x1, y1, x2, y2],
                       fill=(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)))
    img.save(path, "JPEG", quality=85)


def main():
    scale = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    rng = random.Random(42)
    result = {"scale": scale}

    with tempfile.TemporaryDirectory(prefix=f"perftest_{scale}_") as td:
        lib = Path(td) / "library"
        lib.mkdir()
        index_file = Path(td) / "index.npz"

        # 生成 scale 张素材
        for i in range(scale):
            make_image(lib / f"img_{i:05d}.jpg", rng)
        query = Path(td) / "query.jpg"
        make_image(query, rng)

        result["mem_before_mb"] = round(rss_mb(), 1)

        # ① 全量建索引
        t0 = time.perf_counter()
        update_index(library=lib, index_file=index_file, rebuild=True)
        result["full_build_sec"] = round(time.perf_counter() - t0, 2)
        result["mem_after_index_mb"] = round(rss_mb(), 1)

        # ② 增量更新：新增 10% 图片，只编码新图
        added = scale // 10
        for i in range(scale, scale + added):
            make_image(lib / f"img_{i:05d}.jpg", rng)
        t0 = time.perf_counter()
        update_index(library=lib, index_file=index_file, rebuild=False)
        result["incremental_add_%d_sec" % added] = round(time.perf_counter() - t0, 2)

        # ③ 以图搜图响应时间（编码查询图 + 全量余弦比较 + 排序）
        t0 = time.perf_counter()
        search(query, library=lib, index_file=index_file)
        result["search_sec"] = round(time.perf_counter() - t0, 3)

        # ④ 批量重复分组耗时
        t0 = time.perf_counter()
        generate_groups(library=lib, index_file=index_file, write_report=False)
        result["grouping_sec"] = round(time.perf_counter() - t0, 2)
        result["mem_peak_mb"] = round(rss_mb(), 1)
        result["index_file_mb"] = round(index_file.stat().st_size / 1048576, 2)

    out = ROOT / f"perf_results_{scale}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
