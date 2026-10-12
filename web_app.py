"""图片素材库的本地网页入口：python web_app.py。"""
import argparse
import base64
import csv
import io
import json
import hashlib
import hmac
import re
import secrets
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageOps

from group_duplicates import generate_groups
from incremental_search import ROOT, LIBRARY, INDEX_FILE, EXTENSIONS, Encoder, fingerprint, read_index, save_index, update_index

WEB = ROOT / "web"
MAX_FILE = 20 * 1024 * 1024


def safe_file(folder, name):
    if not isinstance(name, str) or not name or "/" in name or "\\" in name or ":" in name or name in {".", ".."}:
        raise ValueError("无效文件名")
    path = folder / name
    if path.resolve().parent != folder.resolve():
        raise ValueError("文件路径超出素材目录")
    return path


def decode_image(payload):
    if not isinstance(payload, str) or len(payload) > MAX_FILE * 4 // 3 + 8:
        raise ValueError("单张图片不能超过 20 MB")
    try:
        data = base64.b64decode(payload, validate=True)
        if not data or len(data) > MAX_FILE:
            raise ValueError("图片为空或超过 20 MB")
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            image.load()
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError("无法读取图片，请使用有效的 JPG、PNG、WebP 或 BMP 图片") from error
    return data


@lru_cache(maxsize=160)
def thumbnail(path, modified, size):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((640, 480), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=82)
        return buffer.getvalue()


class LibraryApp:
    def __init__(self, library=LIBRARY, index=INDEX_FILE):
        self.library, self.index = library, index
        self.library.mkdir(parents=True, exist_ok=True)
        self.token = secrets.token_urlsafe(32)
        self.encoder = Encoder()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.lock = threading.RLock()
        self.jobs = {}
        self.active_job = None
        self.working = False
        self.metadata_cache = {}

    def snapshot(self):
        records, _ = read_index(self.index)
        items = []
        for path in sorted(self.library.iterdir()):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            stat = None
            try:
                stat = path.stat()
                key = (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
                cached = self.metadata_cache.get(path.name)
                if cached is None or cached[0] != key:
                    with Image.open(path) as image:
                        # EXIF 转向 5～8 会交换宽高，无需解码整张图。
                        width, height = image.size
                        if image.getexif().get(274) in (5, 6, 7, 8):
                            width, height = height, width
                    cached = (key, width, height, fingerprint(path))
                    self.metadata_cache[path.name] = cached
                _, width, height, digest = cached
                status = "ready" if records.get(path.name, {}).get("sha256") == digest else "pending"
                error = None
            except FileNotFoundError:
                continue
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                if stat is None:
                    continue
                width = height = 0
                status, error = "failed", str(exc)
            items.append(dict(name=path.name, width=width, height=height, bytes=stat.st_size,
                              version=stat.st_mtime_ns, status=status, error=error))
        ready = sum(i["status"] == "ready" for i in items)
        return dict(items=items, total=len(items), ready=ready, pending=len(items) - ready,
                    recoverable=self.recoverable_batches(),
                    active_job=self.active_job if self.working else None)

    def submit(self, action, task):
        with self.lock:
            if self.working:
                raise ValueError("已有任务正在运行，请等待完成后再操作")
            identifier = uuid.uuid4().hex
            self.working = True
            self.active_job = identifier
            self.jobs[identifier] = dict(id=identifier, action=action, state="running", started=time.time())
            # 保留最近 20 个任务。
            for key in list(self.jobs)[:-20]:
                self.jobs.pop(key)

        def run():
            try:
                result = task()
                with self.lock:
                    self.jobs[identifier].update(state="done", result=result)
            except Exception as error:
                with self.lock:
                    self.jobs[identifier].update(state="failed", error=str(error))
            finally:
                with self.lock:
                    self.working = False

        self.executor.submit(run)
        return {"job": identifier}

    def upload(self, body):
        with self.lock:
            if self.working:
                raise ValueError("任务运行期间暂不能导入，请稍后重试")
            original = body.get("name", "")
            name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", original).strip(" .")[:160]
            if not name or Path(name).suffix.lower() not in EXTENSIONS:
                raise ValueError("请选择 JPG、PNG、WebP 或 BMP 图片")
            if Path(name).stem.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
                name = "image_" + name
            data = decode_image(body.get("data"))
            path = safe_file(self.library, name)
            number = 2
            while path.exists():
                path = safe_file(self.library, f"{Path(name).stem}_{number}{Path(name).suffix}")
                number += 1
            with path.open("xb") as stream:
                stream.write(data)
            return {"name": path.name, "status": "pending"}

    def sync_index(self):
        """搜索、分组前同步变化；未变化时不重新编码。"""
        records, _ = read_index(self.index)
        snapshot = self.snapshot()
        names = {item["name"] for item in snapshot["items"]}
        if snapshot["pending"] or names != set(records):
            return update_index(self.library, self.index, self.encoder)
        return {"failures": {}, "ready": len(records)}

    def query(self, data):
        sync = self.sync_index()
        records, vectors = read_index(self.index)
        if not records:
            raise ValueError("索引为空，请先导入图片并更新索引")
        self.index.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.index.parent) as directory:
            query = Path(directory) / "query"
            query.write_bytes(data)
            vector = self.encoder(query)
        results, skipped = [], 0
        for name, record in records.items():
            path = safe_file(self.library, name)
            if not path.is_file() or fingerprint(path) != record["sha256"]:
                skipped += 1
                continue
            results.append({"name": name, "score": float(np.clip(vectors[name] @ vector, -1, 1))})
        results.sort(key=lambda item: (-item["score"], item["name"]))
        # 返回全量得分供前端做阈值过滤与双阈值对比，一次编码多次复用。
        return {"results": results, "searched": len(results), "skipped": skipped,
                "failures": sync["failures"]}

    def sign_review(self, review):
        encoded = json.dumps(review, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hmac.new(self.token.encode(), encoded, hashlib.sha256).hexdigest()

    def recoverable_batches(self):
        trash = self.index.parent / "trash"
        batches = []
        if trash.exists():
            for path in trash.glob("*/receipt.json"):
                try:
                    receipt = json.loads(path.read_text(encoding="utf-8"))
                    remaining = [name for name in receipt["files"] if safe_file(path.parent / "files", name).is_file()]
                    if remaining:
                        batches.append({"id": path.parent.name, "count": len(remaining), "created": receipt["created"]})
                except (OSError, ValueError, KeyError):
                    continue
        return sorted(batches, key=lambda item: item["created"], reverse=True)

    def cleanup(self, body):
        review, signature = body.get("review"), body.get("signature", "")
        if not isinstance(review, dict) or not isinstance(signature, str) or not hmac.compare_digest(signature, self.sign_review(review)):
            raise ValueError("分组结果已失效，请重新分析后清理")
        names = body.get("names")
        if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
            raise ValueError("请先选择要清理的重复版本")
        names = list(dict.fromkeys(names))
        members = review["hashes"]
        keep = body.get("keep")
        if keep not in members or keep in names or any(name not in members for name in names):
            raise ValueError("清理范围不属于已查看的重复组，或包含要保留的图片")
        records, vectors = read_index(self.index)
        for name, digest in members.items():
            path = safe_file(self.library, name)
            if not path.is_file() or fingerprint(path) != digest or records.get(name, {}).get("sha256") != digest:
                raise ValueError("组内图片已变化，请更新索引并重新分析；本次未清理任何图片")
        batch_id = uuid.uuid4().hex
        batch = self.index.parent / "trash" / batch_id
        storage = batch / "files"
        storage.mkdir(parents=True)
        receipt = {"created": time.time(), "files": {n: records[n]["sha256"] for n in names}, "keep": keep}
        (batch / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False), encoding="utf-8")
        save_index(batch / "vectors.npz", {n: records[n] for n in names}, {n: vectors[n] for n in names})
        moved = []
        try:
            for name in names:
                safe_file(self.library, name).replace(safe_file(storage, name))
                moved.append(name)
            save_index(self.index, {n: r for n, r in records.items() if n not in names},
                       {n: v for n, v in vectors.items() if n not in names})
        except Exception:
            for name in reversed(moved):
                safe_file(storage, name).replace(safe_file(self.library, name))
            raise
        return {"deleted": names, "rejected": [], "batch": batch_id,
                "counts": {"deleted": len(names)}, "message": "已移入本地回收区，可恢复"}

    def restore(self, body):
        identifier = body.get("batch", "")
        if not isinstance(identifier, str) or not re.fullmatch(r"[0-9a-f]{32}", identifier):
            raise ValueError("无效回收批次")
        batch = self.index.parent / "trash" / identifier
        receipt = json.loads((batch / "receipt.json").read_text(encoding="utf-8"))
        old_records, old_vectors = read_index(batch / "vectors.npz")
        records, vectors = read_index(self.index)
        names = [n for n in receipt["files"] if safe_file(batch / "files", n).is_file()]
        if not names:
            raise ValueError("该批次已恢复")
        for name in names:
            if safe_file(self.library, name).exists() or name in records:
                raise ValueError(f"素材库已有同名文件 {name}，为避免覆盖，本次未恢复")
            if fingerprint(safe_file(batch / "files", name)) != receipt["files"][name]:
                raise ValueError("回收文件已变化，停止恢复")
        moved = []
        try:
            for name in names:
                safe_file(batch / "files", name).replace(safe_file(self.library, name))
                moved.append(name)
            save_index(self.index, {**records, **{n: old_records[n] for n in names}},
                       {**vectors, **{n: old_vectors[n] for n in names}})
        except Exception:
            for name in reversed(moved):
                safe_file(self.library, name).replace(safe_file(batch / "files", name))
            raise
        return {"restored": names}

    def groups(self, parameters, sync=True):
        cosine = float(parameters.get("similarity", .94))
        distance = int(parameters.get("distance", 6))
        if not np.isfinite(cosine) or not 0 <= cosine <= 1 or not 0 <= distance <= 64:
            raise ValueError("相似度应在 0～1，哈希距离应在 0～64")
        update = self.sync_index() if sync else {"failures": {}}
        if not self.index.exists():
            raise ValueError("素材库为空，请先导入图片")
        report = generate_groups(self.library, self.index, cosine, distance, write_report=False, skip_invalid=True)
        for group in report["groups"]:
            review = {"thresholds": report["thresholds"],
                      "hashes": {n: report["images"][n]["sha256"] for n in group["members"]}}
            group["review"], group["signature"] = review, self.sign_review(review)
        return {"groups": report["groups"], "singletons": report["singletons"],
                "thresholds": report["thresholds"], "total": len(report["images"]),
                "skipped": {**report["skipped"], **update["failures"]}}

    def compare(self, body):
        update = self.sync_index()
        result = {side: self.groups(body[side], sync=False) for side in ("left", "right")}
        for report in result.values():
            report["skipped"].update(update["failures"])
        return result

    def export(self, names):
        if not isinstance(names, list) or not names:
            raise ValueError("请先选择要导出的图片")
        snapshot = {item["name"]: item for item in self.snapshot()["items"]}
        if any(not isinstance(name, str) or name not in snapshot for name in names):
            raise ValueError("选中的图片已不存在，请刷新素材库")
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(["文件名", "宽度", "高度", "文件字节数", "入库状态"])
        for name in dict.fromkeys(names):
            item = snapshot[name]
            # 防止文件名被电子表格当作公式执行。
            csv_name = "'" + name if name.startswith(("=", "+", "-", "@", "\t", "\r")) else name
            writer.writerow([csv_name, item["width"], item["height"], item["bytes"], item["status"]])
        return output.getvalue().encode("utf-8-sig")


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def respond(self, data, content_type="application/json; charset=utf-8", status=200, extra=None):
            if isinstance(data, (dict, list)):
                data = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        def host_valid(self):
            return self.headers.get("Host") in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

        def do_GET(self):
            try:
                if not self.host_valid():
                    return self.respond({"error": "仅允许本机访问"}, status=403)
                url = urlparse(self.path)
                if url.path == "/":
                    page = (WEB / "index.html").read_text(encoding="utf-8").replace("__TOKEN__", app.token)
                    return self.respond(page.encode("utf-8"), "text/html; charset=utf-8")
                if url.path in {"/characters/rilakkuma.webp", "/characters/korilakkuma.webp", "/characters/kiiroitori.webp", "/characters/group.webp"}:
                    return self.respond((WEB / url.path.lstrip("/")).read_bytes(), "image/webp")
                if url.path in {"/app.js", "/style.css", "/rilakkuma.css", "/bear-family.svg"}:
                    path = WEB / url.path[1:]
                    kind = {".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml"}[path.suffix]
                    return self.respond(path.read_bytes(), kind + "; charset=utf-8")
                if url.path == "/api/library":
                    return self.respond(app.snapshot())
                if url.path.startswith("/api/jobs/"):
                    with app.lock:
                        job = dict(app.jobs.get(url.path.rsplit("/", 1)[1], {}))
                    return self.respond(job or {"error": "任务不存在，请重新操作"}, status=200 if job else 404)
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
            try:
                origin = self.headers.get("Origin")
                allowed = {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}
                if not self.host_valid() or self.headers.get("X-App-Token") != app.token or (origin and origin not in allowed):
                    return self.respond({"error": "页面授权失效，请刷新页面"}, status=403)
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length < 30 * 1024 * 1024:
                    raise ValueError("请求为空或过大，单张图片上限 20 MB")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("请求格式错误")
                if self.path == "/api/import":
                    return self.respond(app.upload(body))
                if self.path == "/api/update":
                    return self.respond(app.submit("update", lambda: update_index(app.library, app.index, app.encoder)))
                if self.path == "/api/search":
                    data = decode_image(body.get("data"))
                    return self.respond(app.submit("search", lambda: app.query(data)))
                if self.path == "/api/groups":
                    return self.respond(app.submit("groups", lambda: app.groups(body)))
                if self.path == "/api/compare":
                    return self.respond(app.submit("compare", lambda: app.compare(body)))
                if self.path == "/api/cleanup":
                    return self.respond(app.submit("cleanup", lambda: app.cleanup(body)))
                if self.path == "/api/restore":
                    return self.respond(app.submit("restore", lambda: app.restore(body)))
                if self.path == "/api/export":
                    return self.respond(app.export(body.get("names")), "text/csv; charset=utf-8", extra={"Content-Disposition": 'attachment; filename="image-list.csv"'})
                self.respond({"error": "操作不存在"}, status=404)
            except (ValueError, OSError, KeyError, TypeError) as error:
                self.respond({"error": str(error)}, status=400)
    return Handler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    app = LibraryApp()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    print(f"图片素材库已启动：http://127.0.0.1:{server.server_port}\n保持终端开启；按 Ctrl+C 停止。", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.executor.shutdown(wait=True)
