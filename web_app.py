"""
图片素材库 Web 应用模块

提供基于本地 HTTP 服务器的图片素材库管理功能，包括：
- 图片上传与导入
- 图片索引更新与相似图片搜索
- 重复图片分组分析与清理
- 图片元数据查询与导出
- 回收站恢复功能

使用方法：
    python web_app.py [--port PORT]

默认端口为 8765，启动后在浏览器访问 http://127.0.0.1:8765

主要组件：
- LibraryApp: 核心业务逻辑类，管理素材库、索引、任务队列
- Handler: HTTP 请求处理器，提供 RESTful API
- safe_file: 安全文件路径验证
- decode_image: 图片解码与验证
- thumbnail: 缩略图生成（带缓存）

依赖模块：
- group_duplicates: 重复图片分组算法
- incremental_search: 增量索引与向量搜索
- PIL (Pillow): 图片处理
- numpy: 向量计算

安全特性：
- 仅允许本地访问（127.0.0.1/localhost）
- Token 认证防止 CSRF 攻击
- 文件名安全过滤防止路径遍历
- Content-Security-Policy 防止 XSS
"""
import argparse
import base64
import csv
import io
import json
import platform
import re
import secrets
import sys
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

LOCAL_SITE_PACKAGES = Path(__file__).resolve().parent / ".runtime" / "site-packages"
if LOCAL_SITE_PACKAGES.is_dir():
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))

import numpy as np
from PIL import Image, ImageOps

from group_duplicates import generate_groups
from incremental_search import (
    ROOT, LIBRARY, INDEX_FILE, EXTENSIONS, MODEL, WEIGHTS, Encoder,
    fingerprint, read_index, save_index, update_index,
)
from runtime_compat import LANCZOS
from metadata_store import MetadataStore
from vector_index import backend_name, rank_persistent
from backend_core.recycle_service import RecycleBinService

# Web 静态资源目录，存放前端 HTML、JS、CSS 等文件
WEB = ROOT / "web"
# 单张图片文件大小上限：20 MB
MAX_FILE = 20 * 1024 * 1024


def safe_file(folder, name):
    """
    验证并构造安全的文件路径，防止路径遍历攻击。

    参数:
        folder (Path): 目标文件夹路径，文件必须位于此目录内
        name (str): 文件名，不允许包含路径分隔符或特殊字符

    返回:
        Path: 验证通过后的完整文件路径

    异常:
        ValueError: 当文件名包含非法字符或路径超出目标目录时抛出

    安全特性:
        - 禁止文件名包含 / \\ : 等路径分隔符
        - 禁止 . 和 .. 等特殊目录名
        - 验证解析后的路径仍在目标文件夹内，防止符号链接攻击
    """
    # 检查文件名是否包含非法字符或特殊目录名
    if not isinstance(name, str) or not name or "/" in name or "\\" in name or ":" in name or name in {".", ".."}:
        raise ValueError("无效文件名")
    path = folder / name
    # 验证解析后的绝对路径仍在目标文件夹内，防止路径遍历攻击
    if path.resolve().parent != folder.resolve():
        raise ValueError("文件路径超出素材目录")
    return path


def decode_image(payload):
    """
    解码并验证 Base64 编码的图片数据。

    参数:
        payload (str): Base64 编码的图片字符串

    返回:
        bytes: 解码后的原始图片二进制数据

    异常:
        ValueError: 当数据格式错误、超过大小限制或不是有效图片时抛出

    验证流程:
        1. 检查 Base64 字符串长度是否超过限制（考虑编码膨胀）
        2. Base64 解码并检查二进制大小
        3. 使用 PIL 验证图片格式（verify）
        4. 完整加载图片以检测损坏（load）

    支持的格式:
        JPG、PNG、WebP、BMP 等 PIL 支持的图片格式
    """
    # 快速检查 Base64 字符串长度，MAX_FILE * 4 // 3 + 8 考虑了 Base64 编码膨胀
    if not isinstance(payload, str) or len(payload) > MAX_FILE * 4 // 3 + 8:
        raise ValueError("单张图片不能超过 20 MB")
    try:
        data = base64.b64decode(payload, validate=True)
        if not data or len(data) > MAX_FILE:
            raise ValueError("图片为空或超过 20 MB")
        # 第一次打开：verify() 检查图片格式和头部信息，不加载像素数据
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        # 第二次打开：load() 完整加载像素，检测损坏或压缩炸弹
        with Image.open(io.BytesIO(data)) as image:
            image.load()
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError("无法读取图片，请使用有效的 JPG、PNG、WebP 或 BMP 图片") from error
    return data


@lru_cache(maxsize=160)
def thumbnail(path, modified, size):
    """
    生成图片缩略图，带 LRU 缓存以提高性能。

    参数:
        path (str): 图片文件的完整路径
        modified (int): 文件修改时间（纳秒时间戳），用于缓存失效判断
        size (int): 文件大小（字节），用于缓存键

    返回:
        bytes: JPEG 格式的缩略图二进制数据

    缓存策略:
        - 使用 lru_cache 缓存最近 160 个缩略图
        - 缓存键包含路径、修改时间和大小，任一变化则重新生成
        - 适合前端列表页快速预览，避免重复处理

    处理流程:
        1. 根据 EXIF 信息自动旋转图片方向
        2. 转换为 RGB 模式（去除 alpha 通道）
        3. 使用高质量 Lanczos 算法缩放至 640x480 以内
        4. 以 82% 质量保存为 JPEG 格式
    """
    with Image.open(path) as image:
        # exif_transpose 根据 EXIF Orientation 标签自动旋转，保证方向正确
        image = ImageOps.exif_transpose(image).convert("RGB")
        # thumbnail 保持宽高比缩放，LANCZOS 提供高质量重采样
        image.thumbnail((640, 480), LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=82)
        return buffer.getvalue()


class LibraryApp:
    """
    图片素材库核心应用类。

    管理素材库文件、索引数据、后台任务和缓存，提供完整的业务逻辑接口。

    主要功能:
        - snapshot(): 获取素材库当前状态快照
        - submit(): 提交异步任务到后台执行
        - upload(): 上传图片到素材库
        - query(): 以图搜图，返回相似图片列表
        - groups(): 分析重复图片分组
        - cleanup(): 清理重复图片（移入回收站）
        - restore(): 从回收站恢复图片
        - export(): 导出图片元数据为 CSV

    属性:
        library (Path): 素材库目录路径
        index (Path): 索引文件路径（存储图片特征向量）
        token (str): 安全令牌，用于 CSRF 防护
        encoder (Encoder): 图片特征提取器，用于向量搜索
        executor (ThreadPoolExecutor): 单线程后台任务执行器
        lock (RLock): 线程锁，保护共享状态
        jobs (dict): 任务历史记录，保留最近 20 个
        active_job (str|None): 当前正在执行的任务 ID
        working (bool): 是否有任务正在运行
        metadata_cache (dict): 图片元数据缓存（宽高、指纹）
    """

    def __init__(self, library=LIBRARY, index=INDEX_FILE):
        """
        初始化素材库应用。

        参数:
            library (Path, optional): 素材库目录路径，默认为 LIBRARY 常量
            index (Path, optional): 索引文件路径，默认为 INDEX_FILE 常量

        初始化过程:
            1. 创建素材库目录（如不存在）
            2. 生成随机安全令牌（32 字节 URL 安全字符串）
            3. 初始化图片编码器
            4. 创建单线程池用于后台任务
            5. 初始化线程锁和任务状态
        """
        self.library, self.index = library, index
        # 确保素材库目录存在
        self.library.mkdir(parents=True, exist_ok=True)
        # 生成安全令牌，用于前端请求的 CSRF 防护
        self.token = secrets.token_urlsafe(32)
        # 图片特征编码器，将图片转换为向量用于相似搜索
        self.encoder = Encoder()
        # 单线程执行器，确保任务串行执行避免并发冲突
        self.executor = ThreadPoolExecutor(max_workers=1)
        # 可重入锁，保护 jobs、working 等共享状态
        self.lock = threading.RLock()
        # 任务历史记录，键为任务 ID，值为任务详情
        self.jobs = {}
        # 当前正在执行的任务 ID，无任务时为 None
        self.active_job = None
        # 标记是否有任务正在运行
        self.working = False
        # 图片元数据缓存，键为文件名，值为 (key, width, height, digest)
        self.metadata_cache = {}
        self.metadata = MetadataStore(self.index.with_suffix(".sqlite3"))
        self.recycle = RecycleBinService(
            self.library,
            self.index,
            self.token,
            read_index=lambda path: read_index(path),
            save_index=lambda path, records, vectors: save_index(path, records, vectors),
            metadata=self.metadata,
        )
        self.group_metadata = {}
        # 兼容升级前已存在的 NPZ 索引：首次启动时直接补齐 SQLite 目录，
        # 不重新加载模型或编码图片。后续更新由 update_index 持续同步。
        if self.index.is_file() and self.metadata.stats()["active"] == 0:
            try:
                records, _ = read_index(self.index)
                if records:
                    self.metadata.sync(self.library, records)
            except (OSError, ValueError, RuntimeError):
                # 索引损坏信息由 /api/health 统一报告，不能阻止服务启动。
                pass

    def health(self):
        """Return deployment, model and index diagnostics without loading weights."""

        index_error = None
        try:
            records, vectors = read_index(self.index)
            dimension = len(next(iter(vectors.values()))) if vectors else 512
        except (OSError, ValueError, RuntimeError) as error:
            records, dimension, index_error = {}, None, str(error)
        model = getattr(self.encoder, "model", None)
        if model is None:
            device, loaded = "not-loaded", False
        else:
            loaded = True
            try:
                device = str(next(model.parameters()).device)
            except (AttributeError, StopIteration):
                device = "unknown"
        return {
            "status": "degraded" if index_error else "ok",
            "runtime": {"python": platform.python_version(), "platform": platform.platform()},
            "model": {"name": MODEL, "weights": WEIGHTS, "device": device,
                      "loaded": loaded, "dimension": dimension},
            "index": {"path": str(self.index.resolve()), "exists": self.index.is_file(),
                      "ready": len(records), "error": index_error,
                      "search_backend": backend_name()},
            "library": {"path": str(self.library.resolve()), "exists": self.library.is_dir()},
            "metadata": self.metadata.stats(),
        }

    def snapshot(self):
        """
        获取素材库当前状态快照，包含所有图片的元数据和索引状态。

        返回:
            dict: 包含以下字段的字典：
                - items (list): 图片列表，每项包含 name, width, height, bytes, version, status, error
                - total (int): 图片总数
                - ready (int): 已索引的图片数
                - pending (int): 待索引的图片数
                - recoverable (list): 可恢复的回收批次列表
                - active_job (str|None): 当前正在执行的任务 ID

        处理流程:
            1. 读取索引文件获取已记录的图片信息
            2. 遍历素材库目录，收集所有支持的图片文件
            3. 对每个文件：
               - 获取文件状态（大小、修改时间）
               - 从缓存或图片文件读取宽高信息
               - 计算文件指纹（SHA256）
               - 判断索引状态（ready/pending/failed）
            4. 统计汇总并返回

        性能优化:
            - 使用 metadata_cache 缓存图片元数据，避免重复读取
            - 缓存键包含 (mtime_ns, ctime_ns, size)，文件变化时自动失效
            - EXIF 方向标签 5-8 会交换宽高，无需完整解码即可判断
        """
        # 读取索引文件，records 为 {文件名: {sha256, ...}} 字典
        records, _ = read_index(self.index)
        items = []
        # 按文件名排序遍历素材库
        for path in sorted(self.library.iterdir()):
            # 跳过非文件或不支持的扩展名
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            stat = None
            try:
                stat = path.stat()
                # 缓存键：修改时间、创建时间、文件大小，任一变化则缓存失效
                key = (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
                cached = self.metadata_cache.get(path.name)
                # 缓存未命中或已过期时，重新读取图片元数据
                if cached is None or cached[0] != key:
                    with Image.open(path) as image:
                        # 读取图片尺寸，EXIF 方向标签 274（Orientation）
                        # 值 5-8 表示图片需要旋转 90/270 度，宽高会交换
                        width, height = image.size
                        if image.getexif().get(274) in (5, 6, 7, 8):
                            width, height = height, width
                    # 计算文件指纹（SHA256）用于索引状态判断
                    cached = (key, width, height, fingerprint(path))
                    self.metadata_cache[path.name] = cached
                _, width, height, digest = cached
                # 判断索引状态：索引中存在且 SHA256 匹配则为 ready，否则为 pending
                status = "ready" if records.get(path.name, {}).get("sha256") == digest else "pending"
                error = None
            except FileNotFoundError:
                # 文件可能在遍历过程中被删除，跳过
                continue
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                # 图片读取失败（损坏、格式错误等）
                if stat is None:
                    continue
                width = height = 0
                status, error = "failed", str(exc)
            items.append(dict(name=path.name, width=width, height=height, bytes=stat.st_size,
                              version=stat.st_mtime_ns, status=status, error=error))
        # 统计各状态数量
        ready = sum(i["status"] == "ready" for i in items)
        return dict(items=items, total=len(items), ready=ready, pending=len(items) - ready,
                    recoverable=self.recoverable_batches(),
                    active_job=self.active_job if self.working else None)

    def submit(self, action, task):
        """
        提交异步任务到后台线程池执行。

        参数:
            action (str): 任务类型标识（如 "update", "search", "groups"）
            task (callable): 无参数可调用对象，返回任务结果

        返回:
            dict: 包含 job 字段的字典，值为任务 ID（32 位十六进制字符串）

        异常:
            ValueError: 当已有任务正在运行时抛出

        任务生命周期:
            1. 提交时生成唯一 ID，状态设为 "running"
            2. 任务完成后状态更新为 "done"，保存结果
            3. 任务失败时状态更新为 "failed"，保存错误信息
            4. 保留最近 20 个任务记录，更早的自动清理

        并发控制:
            - 使用单线程池确保任务串行执行
            - working 标志防止并发提交
            - 使用 RLock 保护共享状态
        """
        with self.lock:
            # 同一时间只允许一个任务运行
            if self.working:
                raise ValueError("已有任务正在运行，请等待完成后再操作")
            # 生成 32 位十六进制任务 ID
            identifier = uuid.uuid4().hex
            self.working = True
            self.active_job = identifier
            self.jobs[identifier] = dict(
                id=identifier, action=action, state="running", started=time.time(),
                progress={"current": 0, "total": 0, "percent": 0, "name": None},
            )
            # 保留最近 20 个任务记录，清理更早的历史
            for key in list(self.jobs)[:-20]:
                self.jobs.pop(key)

        def run():
            """后台执行任务并更新状态"""
            try:
                result = task()
                with self.lock:
                    self.jobs[identifier].update(state="done", result=result)
            except Exception as error:
                traceback.print_exc()
                message = str(error).strip() or (
                    f"{type(error).__name__}（异常未提供详细信息，请查看服务日志）"
                )
                with self.lock:
                    self.jobs[identifier].update(state="failed", error=message)
            finally:
                with self.lock:
                    self.working = False

        # 提交到线程池异步执行
        self.executor.submit(run)
        return {"job": identifier}

    def update_progress(self, current, total, name, status):
        """Publish resumable indexing progress to the active background job."""

        with self.lock:
            identifier = self.active_job
            if identifier in self.jobs:
                self.jobs[identifier]["progress"] = {
                    "current": current, "total": total,
                    "percent": round(current * 100 / total, 1) if total else 100,
                    "name": name, "status": status,
                }

    def update_index_task(self, rebuild=False):
        return update_index(
            self.library, self.index, self.encoder, rebuild=rebuild,
            progress=self.update_progress,
        )

    def upload(self, body):
        """
        上传图片到素材库。

        参数:
            body (dict): 包含以下字段：
                - name (str): 原始文件名
                - data (str): Base64 编码的图片数据

        返回:
            dict: 包含 name（保存后的文件名）和 status（"pending"）的字典

        异常:
            ValueError: 当文件名无效、格式不支持或任务运行中时抛出

        文件名处理:
            1. 过滤非法字符（<>:"/\\|?* 和控制字符）
            2. 去除首尾空格和点号，截断至 160 字符
            3. 检查是否为 Windows 保留设备名（CON、PRN 等）
            4. 文件重名时自动添加数字后缀（_2、_3...）

        安全限制:
            - 任务运行期间禁止上传，避免文件变化导致索引不一致
            - 使用 "xb" 模式独占创建，防止覆盖现有文件
        """
        with self.lock:
            # 任务运行期间禁止上传，避免文件变化导致索引不一致
            if self.working:
                raise ValueError("任务运行期间暂不能导入，请稍后重试")
            original = body.get("name", "")
            # 过滤非法字符，替换为下划线
            name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", original).strip(" .")[:160]
            # 验证文件名和扩展名
            if not name or Path(name).suffix.lower() not in EXTENSIONS:
                raise ValueError("请选择 JPG、PNG、WebP 或 BMP 图片")
            # Windows 保留设备名检查，避免跨平台问题
            if Path(name).stem.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
                name = "image_" + name
            # 解码并验证图片数据
            data = decode_image(body.get("data"))
            path = safe_file(self.library, name)
            # 文件重名时自动添加数字后缀
            number = 2
            while path.exists():
                path = safe_file(self.library, f"{Path(name).stem}_{number}{Path(name).suffix}")
                number += 1
            # "xb" 模式：独占创建，文件已存在时失败
            with path.open("xb") as stream:
                stream.write(data)
            return {"name": path.name, "status": "pending"}

    def sync_index(self):
        """搜索、分组前同步变化；未变化时不重新编码。"""
        records, _ = read_index(self.index)
        snapshot = self.snapshot()
        names = {item["name"] for item in snapshot["items"]}
        if snapshot["pending"] or names != set(records):
            return update_index(self.library, self.index, self.encoder,
                                progress=self.update_progress)
        if self.metadata.stats()["active"] != len(records):
            self.metadata.sync(self.library, records)
        return {"failures": {}, "ready": len(records)}

    def query(self, data):
        """
        以图搜图：根据查询图片在索引中搜索相似图片。

        参数:
            data (bytes): 查询图片的二进制数据

        返回:
            dict: 包含以下字段：
                - results (list): 相似图片列表，每项包含 name 和 score（余弦相似度）
                - searched (int): 搜索到的图片数量
                - skipped (int): 跳过的图片数量（文件不存在或指纹不匹配）

        异常:
            ValueError: 当索引为空时抛出

        搜索流程:
            1. 读取索引文件获取所有图片的记录和向量
            2. 将查询图片写入临时文件，提取特征向量
            3. 遍历索引，计算查询向量与每个图片向量的余弦相似度
            4. 按相似度降序排序，返回全量结果

        性能说明:
            - 一次编码多次复用，前端可自行过滤阈值
            - 使用 @ 运算符进行向量点积（numpy 优化）
            - np.clip 将相似度限制在 [-1, 1] 范围内
        """
        # 读取索引：records 为 {文件名: {sha256, ...}}，vectors 为 {文件名: 向量}
        sync = self.sync_index()
        records, vectors = read_index(self.index)
        if not records:
            raise ValueError("索引为空，请先导入图片并更新索引")
        # 确保索引目录存在
        self.index.parent.mkdir(parents=True, exist_ok=True)
        # 将查询图片写入临时文件，避免占用内存
        with tempfile.TemporaryDirectory(dir=self.index.parent) as directory:
            query = Path(directory) / "query"
            query.write_bytes(data)
            # 提取查询图片的特征向量
            vector = self.encoder(query)
        valid_records, valid_vectors, skipped = {}, {}, 0
        # 遍历索引中的每张图片
        for name, record in records.items():
            path = safe_file(self.library, name)
            # 跳过文件不存在或指纹不匹配的记录（文件已变化）
            if not path.is_file() or fingerprint(path) != record["sha256"]:
                skipped += 1
                continue
            valid_records[name], valid_vectors[name] = record, vectors[name]
        ranked, search_backend = rank_persistent(
            self.index, valid_records, valid_vectors, vector
        )
        results = [{"name": name, "score": float(np.clip(score, -1, 1))}
                   for name, score in ranked]
        results.sort(key=lambda item: (-item["score"], item["name"]))
        # 返回全量得分供前端做阈值过滤与双阈值对比，一次编码多次复用。
        return {"results": results, "searched": len(results), "skipped": skipped,
                "backend": search_backend,
                "failures": sync["failures"]}

    def sign_review(self, review):
        """Sign a duplicate-group review with the personal recycle service."""

        return self.recycle.sign_review(review)

    def recoverable_batches(self):
        """List recoverable soft-delete batches."""

        return self.recycle.recoverable_batches()

    def cleanup(self, body):
        """Soft-delete reviewed duplicates through the transactional service."""

        return self.recycle.cleanup(body)

    def restore(self, body):
        """Restore one recycle batch without overwriting library files."""

        return self.recycle.restore(body)

    def groups(self, parameters, sync=True):
        """
        分析重复图片分组。

        参数:
            parameters (dict): 包含以下可选参数：
                - similarity (float): 余弦相似度阈值，默认 0.94，范围 [0, 1]
                - distance (int): 感知哈希距离阈值，默认 6，范围 [0, 64]

        返回:
            dict: 包含以下字段：
                - groups (list): 重复分组列表，每组包含 members、review、signature
                - singletons (int): 无重复的图片数量
                - thresholds (dict): 使用的阈值参数
                - total (int): 参与分析的图片总数

        异常:
            ValueError: 当参数超出有效范围时抛出

        分组算法:
            1. 调用 generate_groups 执行分组逻辑
            2. 为每个分组生成审查数据和签名
            3. 前端使用签名验证后才能执行清理操作
        """
        # 解析参数，提供默认值
        cosine = float(parameters.get("similarity", .94))
        distance = int(parameters.get("distance", 6))
        # 验证参数范围
        if not np.isfinite(cosine) or not 0 <= cosine <= 1 or not 0 <= distance <= 64:
            raise ValueError("相似度应在 0～1，哈希距离应在 0～64")
        update = self.sync_index() if sync else {"failures": {}}
        if not self.index.exists():
            raise ValueError("素材库为空，请先导入图片")
        report = generate_groups(self.library, self.index, cosine, distance, write_report=False, skip_invalid=True)
        for group in report["groups"]:
            review = {"thresholds": report["thresholds"],
                      "hashes": {n: report["images"][n]["sha256"] for n in group["members"]}}
            # 签名用于清理时的安全验证
            group["review"], group["signature"] = review, self.sign_review(review)
            for name in group["members"]:
                self.group_metadata[name] = {
                    "group_id": group["id"], "category": group["category"],
                    "quality_score": group.get("quality_scores", {}).get(name),
                    "suggested_keep": name == group["suggested_keep"],
                }
        return {"groups": report["groups"], "singletons": report["singletons"],
                "thresholds": report["thresholds"], "total": len(report["images"]),
                "candidate_backend": report.get("candidate_backend"),
                "candidate_pair_count": report.get("candidate_pair_count", 0),
                "similar_pair_count": report.get("similar_pair_count", 0),
                "skipped": {**report["skipped"], **update["failures"]}}

    def compare(self, body):
        update = self.sync_index()
        result = {side: self.groups(body[side], sync=False) for side in ("left", "right")}
        for report in result.values():
            report["skipped"].update(update["failures"])
        return result

    def export(self, names):
        """
        导出图片元数据为 CSV 格式。

        参数:
            names (list): 要导出的文件名列表

        返回:
            bytes: UTF-8 BOM 编码的 CSV 数据

        异常:
            ValueError: 当参数为空或包含不存在的文件时抛出

        CSV 格式:
            - 列：文件名、宽度、高度、文件字节数、入库状态
            - 编码：UTF-8 with BOM（便于 Excel 识别）
            - 安全处理：文件名以公式字符开头时添加单引号前缀

        安全措施:
            - 防止 CSV 注入攻击（公式注入）
            - 去重处理，保持原始顺序
        """
        # 验证参数
        if not isinstance(names, list) or not names:
            raise ValueError("请先选择要导出的图片")
        # 获取当前素材库快照
        snapshot = {item["name"]: item for item in self.snapshot()["items"]}
        # 验证所有文件名有效
        if any(not isinstance(name, str) or name not in snapshot for name in names):
            raise ValueError("选中的图片已不存在，请刷新素材库")
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        # 写入表头
        catalogue = self.metadata.by_names(names)
        writer.writerow(["稳定ID", "文件名", "宽度", "高度", "格式", "文件字节数", "SHA-256",
                         "入库状态", "重复组", "重复类别", "质量分", "推荐保留", "模型", "权重"])
        # 去重，保持顺序
        for name in dict.fromkeys(names):
            item = snapshot[name]
            # 防止 CSV 注入：文件名以公式字符开头时添加单引号前缀
            # 避免电子表格软件将文件名当作公式执行
            csv_name = "'" + name if name.startswith(("=", "+", "-", "@", "\t", "\r")) else name
            metadata = catalogue.get(name, {})
            group = self.group_metadata.get(name, {})
            writer.writerow([metadata.get("id", ""), csv_name, item["width"], item["height"],
                             metadata.get("format", ""), item["bytes"], metadata.get("sha256", ""),
                             item["status"], group.get("group_id", ""), group.get("category", ""),
                             group.get("quality_score", ""), group.get("suggested_keep", ""), MODEL, WEIGHTS])
        # 返回 UTF-8 BOM 编码的字节数据
        return output.getvalue().encode("utf-8-sig")

    def export_json(self, names):
        """Export the same auditable metadata as structured JSON."""

        if not isinstance(names, list) or not names:
            raise ValueError("请先选择要导出的图片")
        snapshot = {item["name"]: item for item in self.snapshot()["items"]}
        if any(not isinstance(name, str) or name not in snapshot for name in names):
            raise ValueError("选中的图片已不存在，请刷新素材库")
        catalogue = self.metadata.by_names(names)
        rows = []
        for name in dict.fromkeys(names):
            rows.append({**snapshot[name], **catalogue.get(name, {}),
                         **self.group_metadata.get(name, {}),
                         "model": MODEL, "weights": WEIGHTS})
        return json.dumps({"exported_at": time.time(), "items": rows},
                          ensure_ascii=False, indent=2).encode("utf-8")


def make_handler(app):
    """
    创建 HTTP 请求处理器类。

    参数:
        app (LibraryApp): 素材库应用实例

    返回:
        type: Handler 类（BaseHTTPRequestHandler 的子类）

    设计说明:
        - 使用闭包捕获 app 实例，避免全局变量
        - 每次调用返回新的类，支持多实例隔离
        - Handler 类实现 RESTful API 路由

    API 端点:
        GET:
            / - 主页 HTML
            /api/library - 素材库快照
            /api/jobs/<id> - 任务状态查询
            /api/thumbnail?name=<file> - 缩略图
        POST:
            /api/import - 上传图片
            /api/update - 更新索引
            /api/search - 以图搜图
            /api/groups - 分析重复分组
            /api/compare - 对比两组分组结果
            /api/cleanup - 清理重复图片
            /api/restore - 恢复回收站文件
            /api/export - 导出 CSV
    """
    class Handler(BaseHTTPRequestHandler):
        """HTTP 请求处理器，提供 Web API 和静态资源服务"""

        def log_message(self, format, *args):
            """禁用默认请求日志输出"""
            pass

        def respond(self, data, content_type="application/json; charset=utf-8", status=200, extra=None):
            """
            发送 HTTP 响应。

            参数:
                data: 响应数据，字典/列表自动序列化为 JSON
                content_type (str): Content-Type 头，默认 application/json
                status (int): HTTP 状态码，默认 200
                extra (dict, optional): 额外的响应头

            安全头:
                - Cache-Control: no-store（禁止缓存）
                - X-Content-Type-Options: nosniff（禁止 MIME 嗅探）
                - Content-Security-Policy: 严格的 CSP 策略防止 XSS
            """
            # 字典/列表自动序列化为 JSON
            if isinstance(data, (dict, list)):
                data = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            # 安全响应头
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            # CSP：仅允许同源资源，防止 XSS 和数据注入
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        def host_valid(self):
            """
            验证 Host 头，确保仅允许本地访问。

            返回:
                bool: Host 头是否为 127.0.0.1 或 localhost
            """
            return self.headers.get("Host") in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

        def do_GET(self):
            """
            处理 GET 请求。

            路由:
                / - 主页 HTML（注入安全令牌）
                /characters/*.webp - 角色图片
                /app.js, /style.css 等 - 静态资源
                /api/library - 素材库快照
                /api/jobs/<id> - 任务状态查询
                /api/thumbnail?name=<file> - 缩略图

            安全:
                - 验证 Host 头，仅允许本地访问
                - 主页注入令牌后返回
            """
            try:
                # 验证 Host 头，防止 DNS 重绑定攻击
                if not self.host_valid():
                    return self.respond({"error": "仅允许本机访问"}, status=403)
                url = urlparse(self.path)
                # 主页：注入安全令牌到 HTML
                if url.path == "/":
                    page = (WEB / "index.html").read_text(encoding="utf-8").replace("__TOKEN__", app.token)
                    return self.respond(page.encode("utf-8"), "text/html; charset=utf-8")
                # 角色图片（固定路径）
                if url.path in {"/characters/rilakkuma.webp", "/characters/korilakkuma.webp", "/characters/kiiroitori.webp", "/characters/group.webp"}:
                    return self.respond((WEB / url.path.lstrip("/")).read_bytes(), "image/webp")
                # 静态资源（JS、CSS、SVG）
                if url.path in {"/app.js", "/style.css", "/rilakkuma.css", "/bear-family.svg"}:
                    path = WEB / url.path[1:]
                    kind = {".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml"}[path.suffix]
                    return self.respond(path.read_bytes(), kind + "; charset=utf-8")
                # API：素材库快照
                if url.path == "/api/library":
                    return self.respond(app.snapshot())
                if url.path == "/api/health":
                    return self.respond(app.health())
                if url.path == "/api/catalog":
                    query = parse_qs(url.query)
                    return self.respond(app.metadata.list(
                        limit=query.get("limit", [50])[0], offset=query.get("offset", [0])[0],
                        status=query.get("status", [None])[0], query=query.get("q", [None])[0],
                    ))
                # API：任务状态查询
                if url.path.startswith("/api/jobs/"):
                    with app.lock:
                        job = dict(app.jobs.get(url.path.rsplit("/", 1)[1], {}))
                    return self.respond(job or {"error": "任务不存在，请重新操作"}, status=200 if job else 404)
                # API：缩略图
                if url.path == "/api/thumbnail":
                    name = parse_qs(url.query).get("name", [""])[0]
                    path = safe_file(app.library, name)
                    if path.suffix.lower() not in EXTENSIONS:
                        raise ValueError("不是支持的图片")
                    stat = path.stat()
                    return self.respond(thumbnail(str(path), stat.st_mtime_ns, stat.st_size), "image/jpeg")
                self.respond({"error": "页面不存在"}, status=404)
            except (ValueError, OSError, RuntimeError, Image.DecompressionBombError) as error:
                self.respond({"error": str(error)}, status=400)

        def do_POST(self):
            """
            处理 POST 请求。

            安全验证:
                1. Host 头验证（本地访问）
                2. X-App-Token 验证（CSRF 防护）
                3. Origin 头验证（可选，防止跨域攻击）

            路由:
                /api/import - 上传图片
                /api/update - 更新索引
                /api/search - 以图搜图
                /api/groups - 分析重复分组
                /api/compare - 对比两组分组结果
                /api/cleanup - 清理重复图片
                /api/restore - 恢复回收站文件
                /api/export - 导出 CSV

            请求格式:
                - Content-Type: application/json
                - Body: JSON 对象
                - 大小限制: 30 MB
            """
            try:
                # CSRF 防护：验证 Host、Token 和 Origin
                origin = self.headers.get("Origin")
                allowed = {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}
                if not self.host_valid() or self.headers.get("X-App-Token") != app.token or (origin and origin not in allowed):
                    return self.respond({"error": "页面授权失效，请刷新页面"}, status=403)
                # 验证请求体大小
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length < 30 * 1024 * 1024:
                    raise ValueError("请求为空或过大，单张图片上限 20 MB")
                # 解析 JSON 请求体
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("请求格式错误")
                # 路由分发
                if self.path == "/api/import":
                    return self.respond(app.upload(body))
                if self.path == "/api/update":
                    # 更新索引：提交异步任务
                    return self.respond(app.submit("update", lambda: app.update_index_task(
                        rebuild=bool(body.get("rebuild", False)))))
                if self.path == "/api/search":
                    # 以图搜图：解码图片后提交异步任务
                    data = decode_image(body.get("data"))
                    return self.respond(app.submit("search", lambda: app.query(data)))
                if self.path == "/api/groups":
                    # 分析重复分组
                    return self.respond(app.submit("groups", lambda: app.groups(body)))
                if self.path == "/api/compare":
                    return self.respond(app.submit("compare", lambda: app.compare(body)))
                if self.path == "/api/cleanup":
                    # 清理重复图片
                    return self.respond(app.submit("cleanup", lambda: app.cleanup(body)))
                if self.path == "/api/restore":
                    # 恢复回收站文件
                    return self.respond(app.submit("restore", lambda: app.restore(body)))
                if self.path == "/api/export":
                    if body.get("format") == "json":
                        return self.respond(app.export_json(body.get("names")), "application/json; charset=utf-8",
                                            extra={"Content-Disposition": 'attachment; filename="image-list.json"'})
                    # 导出 CSV，添加 Content-Disposition 头触发下载
                    return self.respond(app.export(body.get("names")), "text/csv; charset=utf-8", extra={"Content-Disposition": 'attachment; filename="image-list.csv"'})
                self.respond({"error": "操作不存在"}, status=404)
            except (ValueError, OSError, KeyError, TypeError) as error:
                self.respond({"error": str(error)}, status=400)
    return Handler


if __name__ == "__main__":
    """
    命令行入口：解析参数并启动 HTTP 服务器。

    使用方法:
        python web_app.py [--port PORT]

    参数:
        --port (int): HTTP 服务器端口，默认 8765

    启动流程:
        1. 解析命令行参数
        2. 创建 LibraryApp 实例
        3. 创建 ThreadingHTTPServer（多线程支持）
        4. 绑定到 127.0.0.1（仅本地访问）
        5. 启动服务循环，等待 Ctrl+C 退出
        6. 优雅关闭：停止服务器并等待后台任务完成
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    # 创建素材库应用实例
    app = LibraryApp()
    # 创建多线程 HTTP 服务器，绑定到本地回环地址
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    print(f"图片素材库已启动：http://127.0.0.1:{server.server_port}\n保持终端开启；按 Ctrl+C 停止。", flush=True)
    try:
        # 启动服务循环
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        # 优雅关闭：停止服务器并等待后台任务完成
        server.server_close()
        app.executor.shutdown(wait=True)
