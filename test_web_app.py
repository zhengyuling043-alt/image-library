"""Web 应用端到端测试（TC01～TC22，共 22 条，用例设计详见 TEST_CASES.md）。

测什么：进程内真实启动 ThreadingHTTPServer、发真实 HTTP 请求，验证 web_app.py
的每一条路由与业务逻辑——不 mock 接口，测的是接口真身。
用 FakeEncoder（固定单位向量）替代真实 CLIP，不下载模型、不改用户素材，毫秒级跑完。

分组说明：
  导入校验  TC01 路径穿越拒绝 / TC02 超 20MB 拒绝 / TC03 损坏数据拒绝 /
            TC04 合法 PNG 通过 / TC05 不支持格式拒绝 / TC06 危险文件名清洗 /
            TC07 同名文件自动另存
  任务队列  TC08 导入与索引任务串行执行
  参数校验  TC09 非法相似度/哈希距离拒绝
  以图搜图  TC10 空索引返回可读提示
  素材浏览  TC11 快照三态（ready/pending/failed）计数准确
  安全鉴权  TC12 导出 CSV 鉴权 / TC13 伪造 Token 拒绝
  端到端    TC14 导入→建索引→检索全链路
  缩略图    TC15 缩略图生成 / 错误页 404
  清理恢复  TC16 按组清理移出多余版本 / TC17 检索全量排序且得分恰 1.0 /
            TC18 恢复批次 / TC19 签名失效拒绝 / TC20 清理失败自动回滚 /
            TC21 恢复拒绝覆盖同名文件 / TC22 分组被篡改时拒绝清理

运行方式：python -m unittest test_web_app -v
"""
import base64
import io
import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from PIL import Image

import web_app
from incremental_search import fingerprint, save_index


def png_bytes(color=(10, 20, 30), size=(8, 6)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, "PNG")
    return buffer.getvalue()


def png_from_array(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, "PNG")
    return buffer.getvalue()


def b64(data):
    return base64.b64encode(data).decode()


class FakeEncoder:
    """替代编码器：固定单位向量，不访问模型下载平台。"""

    def __call__(self, path):
        vector = np.zeros(512, dtype=np.float32)
        vector[0] = 1.0
        return vector


class WebAppTestCase(unittest.TestCase):
    def test_new_unnumbered_upload_searches_without_manual_update(self):
        data = png_bytes()
        self.app.upload({"name": "假日海边.png", "data": b64(data)})
        result = self.app.query(data)
        self.assertEqual(result["results"][0]["name"], "假日海边.png")
        self.assertAlmostEqual(result["results"][0]["score"], 1.0, places=5)
        self.assertEqual(self.app.snapshot()["ready"], 1)

    def test_groups_and_compare_skip_bad_image_and_sync_new_copies(self):
        data = png_bytes()
        self.app.upload({"name": "原图.png", "data": b64(data)})
        self.app.upload({"name": "副本.png", "data": b64(data)})
        (self.library / "broken.jpg").write_bytes(b"broken")
        original_encoder = self.app.encoder
        def validating_encoder(path):
            with Image.open(path) as image:
                image.load()
            return original_encoder(path)
        self.app.encoder = validating_encoder
        base_url = self.serve()
        for endpoint, body in [("/api/groups", {}), ("/api/compare", {"left": {}, "right": {"similarity": .9, "distance": 18}})]:
            _, response = self.request(base_url, endpoint, body, self.app.token)
            job = self.wait_job(base_url, self.app.token, json.loads(response)["job"])
            self.assertEqual(job["state"], "done", job)
            reports = [job["result"]] if endpoint.endswith("groups") else job["result"].values()
            for report in reports:
                self.assertEqual(report["total"], 2)
                self.assertEqual(set(report["groups"][0]["members"]), {"原图.png", "副本.png"})
                self.assertIn("broken.jpg", report["skipped"])

    def setUp(self):
        patcher = patch.object(web_app, "Encoder", FakeEncoder)
        patcher.start()
        self.addCleanup(patcher.stop)
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.library = self.root / "library"
        self.index = self.root / "index" / "incremental.npz"
        self.app = web_app.LibraryApp(library=self.library, index=self.index)
        self.addCleanup(self.app.executor.shutdown, wait=True)

    # ---------- TC01～TC12：函数与应用层 ----------

    def test_tc01_safe_file_rejects_traversal(self):
        """TC01 路径安全：拒绝穿越与非法名，放行合法名。"""
        for name in ["../x.jpg", "a/b.jpg", "a\\b.jpg", "..", ".", ":x", "", None, "a:b.jpg"]:
            with self.assertRaises(ValueError, msg=name):
                web_app.safe_file(self.library, name)
        path = web_app.safe_file(self.library, "ok.jpg")
        self.assertEqual(path, self.library / "ok.jpg")

    def test_tc02_decode_image_rejects_oversize(self):
        """TC02 图片解码：超过 20 MB 的请求负载直接拒绝。"""
        with self.assertRaises(ValueError):
            web_app.decode_image("A" * (web_app.MAX_FILE * 4 // 3 + 9))

    def test_tc03_decode_image_rejects_corrupt_data(self):
        """TC03 图片解码：非法 base64、损坏图片均拒绝。"""
        with self.assertRaises(ValueError):
            web_app.decode_image("not-base64!!")
        with self.assertRaises(ValueError):
            web_app.decode_image(b64(b"this is not an image"))

    def test_tc04_decode_image_accepts_valid_png(self):
        """TC04 图片解码：有效 PNG 通过并返回原始字节。"""
        data = png_bytes()
        self.assertEqual(web_app.decode_image(b64(data)), data)

    def test_tc05_upload_rejects_unsupported_format(self):
        """TC05 上传校验：不支持的扩展名被拒绝。"""
        data = png_bytes()
        for name in ["page.gif", "notes.txt", "noext"]:
            with self.assertRaises(ValueError, msg=name):
                self.app.upload({"name": name, "data": b64(data)})

    def test_tc06_upload_sanitizes_filename(self):
        """TC06 上传校验：非法字符替换，Windows 保留名加前缀。"""
        data = png_bytes()
        saved = self.app.upload({"name": "co<n>:?.jpg", "data": b64(data)})
        self.assertEqual(saved["name"], "co_n___.jpg")
        reserved = self.app.upload({"name": "CON.jpg", "data": b64(data)})
        self.assertEqual(reserved["name"], "image_CON.jpg")

    def test_tc07_upload_autorenames_duplicates(self):
        """TC07 上传校验：同名文件自动另存为 _2。"""
        data = png_bytes()
        first = self.app.upload({"name": "same.png", "data": b64(data)})
        second = self.app.upload({"name": "same.png", "data": b64(data)})
        self.assertEqual([first["name"], second["name"]], ["same.png", "same_2.png"])
        self.assertTrue((self.library / "same_2.png").is_file())

    def test_tc08_tasks_are_serialized(self):
        """TC08 任务互斥：任务运行期间提交新任务被拒绝。"""
        started, release = threading.Event(), threading.Event()

        def blocker():
            started.set()
            release.wait(timeout=5)

        self.app.submit("block", blocker)
        self.assertTrue(started.wait(timeout=5))
        try:
            with self.assertRaises(ValueError) as caught:
                self.app.submit("other", blocker)
        finally:
            release.set()
        self.assertIn("任务", str(caught.exception))

    def test_tc09_groups_rejects_invalid_thresholds(self):
        """TC09 阈值校验：相似度、哈希距离越界与 NaN 均拒绝。"""
        for body in [{"similarity": -0.1}, {"similarity": 1.5},
                     {"distance": -1}, {"distance": 65},
                     {"similarity": float("nan")}, {"similarity": float("inf")}]:
            with self.assertRaises(ValueError, msg=body):
                self.app.groups({"similarity": .94, "distance": 6, **body})

    def test_tc10_query_requires_index(self):
        """TC10 空索引搜索：给出可读的错误提示。"""
        with self.assertRaises(ValueError) as caught:
            self.app.query(png_bytes())
        self.assertIn("索引为空", str(caught.exception))

    def test_tc11_snapshot_reports_three_states(self):
        """TC11 素材快照：ready / pending / failed 三态判定。"""
        good, changed = png_bytes((1, 2, 3)), png_bytes((4, 5, 6))
        (self.library / "good.png").write_bytes(good)
        (self.library / "changed.jpg").write_bytes(changed)
        (self.library / "broken.png").write_bytes(b"junk")
        records = {"good.png": {"sha256": fingerprint(self.library / "good.png")},
                   "changed.jpg": {"sha256": "stale-hash"}}
        save_index(self.index, records, {n: np.ones(512, np.float32) for n in records})
        items = {item["name"]: item for item in self.app.snapshot()["items"]}
        self.assertEqual(items["good.png"]["status"], "ready")
        self.assertEqual(items["changed.jpg"]["status"], "pending")
        self.assertEqual(items["broken.png"]["status"], "failed")

    def test_tc12_export_csv_and_formula_guard(self):
        """TC12 清单导出：表头、字段正确，危险前缀被转义。"""
        for name in ["ok.jpg", "=evil.jpg"]:
            (self.library / name).write_bytes(png_bytes())
        rows = self.app.export(["=evil.jpg", "ok.jpg", "=evil.jpg"]).decode("utf-8-sig").splitlines()
        self.assertEqual(rows[0], "文件名,宽度,高度,文件字节数,入库状态")
        self.assertEqual(rows[1].split(",")[0], "'=evil.jpg")
        self.assertEqual(rows[2].split(",")[0], "ok.jpg")
        self.assertEqual(len(rows), 3)
        with self.assertRaises(ValueError):
            self.app.export([])
        with self.assertRaises(ValueError):
            self.app.export(["ghost.jpg"])

    # ---------- TC13～TC15：HTTP 接口层 ----------

    def serve(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), web_app.make_handler(self.app))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}"

    def request(self, base_url, path, body=None, token=None, host=None):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["X-App-Token"] = token
        data = None if body is None else json.dumps(body).encode()
        method = "GET" if body is None else "POST"
        req = urllib.request.Request(base_url + path, data=data, headers=headers, method=method)
        if host:
            req.add_header("Host", host)
        try:
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=10) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            with error:
                return error.code, error.read()

    def wait_job(self, base_url, token, job_id, timeout=10):
        deadline = time.time() + timeout
        while time.time() < deadline:
            _, data = self.request(base_url, f"/api/jobs/{job_id}")
            job = json.loads(data)
            if job.get("state") in {"done", "failed"}:
                return job
            time.sleep(0.05)
        raise AssertionError("任务轮询超时")

    def test_tc13_home_token_and_authorization(self):
        """TC13 HTTP 安全：首页注入令牌，未授权与伪造 Host 的 POST 被拒。"""
        base_url = self.serve()
        status, page = self.request(base_url, "/")
        self.assertEqual(status, 200)
        text = page.decode("utf-8")
        self.assertIn(self.app.token, text)
        self.assertNotIn("__TOKEN__", text)
        status, _ = self.request(base_url, "/api/update", body={})
        self.assertEqual(status, 403)
        status, _ = self.request(base_url, "/api/update", body={}, token="wrong-token")
        self.assertEqual(status, 403)
        status, _ = self.request(base_url, "/api/update", body={}, token=self.app.token, host="evil.example:1")
        self.assertEqual(status, 403)

    def test_tc14_import_update_search_end_to_end(self):
        """TC14 全链路：导入 → 更新索引 → 以图搜图。"""
        base_url = self.serve()
        token = self.app.token
        _, body = self.request(base_url, "/api/import", {"name": "sample.png", "data": b64(png_bytes())}, token)
        self.assertEqual(json.loads(body)["status"], "pending")
        _, body = self.request(base_url, "/api/update", {}, token)
        job = self.wait_job(base_url, token, json.loads(body)["job"])
        self.assertEqual(job["result"]["counts"], dict(added=1, modified=0, reused=0, deleted=0, failed=0))
        _, body = self.request(base_url, "/api/library")
        snapshot = json.loads(body)
        self.assertEqual((snapshot["total"], snapshot["ready"]), (1, 1))
        _, body = self.request(base_url, "/api/search", {"data": b64(png_bytes())}, token)
        job = self.wait_job(base_url, token, json.loads(body)["job"])
        self.assertEqual(job["result"]["results"][0]["name"], "sample.png")
        self.assertAlmostEqual(job["result"]["results"][0]["score"], 1.0, places=5)

    def test_tc15_export_thumbnail_and_error_pages(self):
        """TC15 HTTP 接口：导出 CSV（含 BOM）、缩略图、404 与 400。"""
        base_url = self.serve()
        token = self.app.token
        (self.library / "sample.png").write_bytes(png_bytes())
        status, data = self.request(base_url, "/api/export", {"names": ["sample.png"]}, token)
        self.assertEqual(status, 200)
        self.assertTrue(data.startswith(b"\xef\xbb\xbf"))
        self.assertIn("sample.png".encode(), data)
        status, data = self.request(base_url, "/api/thumbnail?name=sample.png")
        self.assertEqual((status, data[:2]), (200, b"\xff\xd8"))
        self.assertEqual(self.request(base_url, "/api/missing")[0], 404)
        self.assertEqual(self.request(base_url, "/api/thumbnail?name=../secret.png")[0], 400)
        self.assertEqual(self.request(base_url, "/api/update", body=[1, 2], token=token)[0], 400)

    def test_tc16_cleanup_removes_extra_versions(self):
        """TC16 扩展·批量清理：只删组内非推荐项，索引同步更新。"""
        base_url = self.serve()
        token = self.app.token
        gradient = np.tile(np.arange(64, dtype=np.uint8) * 4, (64, 1))
        buffer = io.BytesIO()
        Image.fromarray(gradient).resize((32, 32)).save(buffer, "PNG")
        big = png_from_array(gradient)
        small = buffer.getvalue()
        other = png_from_array(gradient.T.copy())
        for name, data in [("master.png", big), ("master_copy.jpg", small), ("single.png", other)]:
            _, body = self.request(base_url, "/api/import", {"name": name, "data": b64(data)}, token)
        _, body = self.request(base_url, "/api/update", {}, token)
        job = self.wait_job(base_url, token, json.loads(body)["job"])
        self.assertEqual(job["result"]["counts"]["added"], 3)
        group = self.app.groups({})["groups"][0]
        cleanup_body = {"names": ["master_copy.jpg"], "keep": "master.png", "review": group["review"], "signature": group["signature"]}
        _, body = self.request(base_url, "/api/cleanup", cleanup_body, token)
        job = self.wait_job(base_url, token, json.loads(body)["job"])
        self.assertEqual(job["result"]["deleted"], ["master_copy.jpg"])
        self.assertEqual(job["result"]["rejected"], [])
        self.assertEqual(job["result"]["counts"]["deleted"], 1)
        self.assertFalse((self.library / "master_copy.jpg").exists())
        self.assertTrue((self.library / "master.png").exists())
        _, body = self.request(base_url, "/api/groups", {"similarity": .94, "distance": 6}, token)
        job = self.wait_job(base_url, token, json.loads(body)["job"])
        self.assertEqual(job["result"]["groups"], [])
        _, body = self.request(base_url, "/api/cleanup", {**cleanup_body, "names": ["master.png", "single.png"]}, token)
        job = self.wait_job(base_url, token, json.loads(body)["job"])
        self.assertEqual(job["state"], "failed")
        self.assertTrue((self.library / "master.png").is_file())
        self.assertTrue((self.library / "single.png").is_file())

    def prepare_duplicate_group(self):
        data = png_bytes()
        for name in ("a.png", "b.png"):
            (self.library / name).write_bytes(data)
        web_app.update_index(self.library, self.index, self.app.encoder)
        group = self.app.groups({})["groups"][0]
        # 人工选择保留非默认推荐项 b.png。
        return {"names": ["a.png"], "keep": "b.png", "review": group["review"], "signature": group["signature"]}

    def test_tc18_manual_keeper_and_restore(self):
        body = self.prepare_duplicate_group()
        result = self.app.cleanup(body)
        self.assertFalse((self.library / "a.png").exists())
        self.assertTrue((self.library / "b.png").is_file())
        self.assertEqual(self.app.snapshot()["ready"], 1)
        self.assertEqual(self.app.recoverable_batches()[0]["id"], result["batch"])
        restored = self.app.restore({"batch": result["batch"]})
        self.assertEqual(restored["restored"], ["a.png"])
        self.assertEqual(self.app.snapshot()["ready"], 2)
        self.assertEqual(self.app.recoverable_batches(), [])

    def test_tc19_changed_picture_invalidates_cleanup(self):
        body = self.prepare_duplicate_group()
        (self.library / "a.png").write_bytes(png_bytes((50, 60, 70)))
        with self.assertRaises(ValueError):
            self.app.cleanup(body)
        self.assertTrue((self.library / "a.png").exists())
        self.assertTrue((self.library / "b.png").exists())

    def test_tc20_cleanup_rolls_back_when_index_save_fails(self):
        body = self.prepare_duplicate_group()
        original = web_app.save_index
        def failing(path, records, vectors):
            if path == self.index:
                raise OSError("simulated disk failure")
            return original(path, records, vectors)
        with patch.object(web_app, "save_index", failing):
            with self.assertRaises(OSError):
                self.app.cleanup(body)
        self.assertEqual(self.app.snapshot()["ready"], 2)
        self.assertEqual(self.app.recoverable_batches(), [])

    def test_tc21_restore_never_overwrites_file(self):
        body = self.prepare_duplicate_group()
        result = self.app.cleanup(body)
        replacement = png_bytes((99, 1, 1))
        (self.library / "a.png").write_bytes(replacement)
        with self.assertRaises(ValueError):
            self.app.restore({"batch": result["batch"]})
        self.assertEqual((self.library / "a.png").read_bytes(), replacement)

    def test_tc22_review_signature_rejects_modified_group(self):
        body = self.prepare_duplicate_group()
        body["review"]["hashes"]["outside.png"] = "fake"
        with self.assertRaises(ValueError):
            self.app.cleanup(body)
        self.assertEqual(self.app.snapshot()["ready"], 2)

    def test_tc17_query_returns_all_sorted_results(self):
        """TC17 扩展·阈值对比数据源：返回全部排序得分，不止前 20。"""
        names = [f"img_{i:02d}.png" for i in range(25)]
        for name in names:
            (self.library / name).write_bytes(png_bytes())
        records = {name: {"sha256": fingerprint(self.library / name)} for name in names}
        # 第 i 张图向量的第 i 维为 1，查询向量前 24 维均为 1/24：前 24 张同分，第 25 张为 0。
        vectors = {name: np.eye(512, dtype=np.float32)[i] for i, name in enumerate(names)}
        save_index(self.index, records, vectors)
        self.app.encoder = lambda path: np.concatenate([np.full(24, 1 / 24, dtype=np.float32), np.zeros(488, dtype=np.float32)])
        result = self.app.query(png_bytes())
        self.assertEqual(result["searched"], 25)
        self.assertEqual(len(result["results"]), 25)
        scores = [row["score"] for row in result["results"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual([row["name"] for row in result["results"][:24]], names[:24])
        self.assertEqual(result["results"][-1]["name"], "img_24.png")


if __name__ == "__main__":
    unittest.main()
