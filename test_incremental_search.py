"""增量索引模块测试（2 条）。

测什么：incremental_search.py 的索引生命周期与容错——
  test_update_lifecycle            增/删/改图片后，索引按"新增/复用/删除/失败"
                                   分类正确更新，失败图片不阻塞其余；
  test_failed_save_preserves_previous_index
                                   索引保存中途失败时，旧的索引文件保持完好
                                   （对应 npz 原子替换设计，防断电损坏）。

运行方式：python -m unittest test_incremental_search -v
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import incremental_search as app


class IncrementalIndexTest(unittest.TestCase):
    def test_update_lifecycle(self):
        # 使用临时文件和替代编码器验证索引逻辑，不下载模型、不改用户图片。
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            library = root / "library"
            library.mkdir()
            index = root / "index.npz"
            calls = []

            def encode(path):
                calls.append(path.name)
                if path.read_bytes() == b"broken":
                    raise OSError("invalid image")
                vector = np.zeros(512, dtype=np.float32)
                vector[0] = 1
                return vector

            def update():
                return app.update_index(library, index, encode)["counts"]

            (library / "a.jpg").write_bytes(b"original")
            self.assertEqual(update()["added"], 1)
            calls.clear()
            self.assertEqual(update()["reused"], 1)
            self.assertEqual(calls, [])
            (library / "b.jpg").write_bytes(b"second")
            self.assertEqual(update()["added"], 1)
            self.assertEqual(calls, ["b.jpg"])
            calls.clear()
            (library / "a.jpg").write_bytes(b"changed")
            self.assertEqual(update()["modified"], 1)
            self.assertEqual(calls, ["a.jpg"])
            (library / "b.jpg").unlink()
            self.assertEqual(update()["deleted"], 1)
            (library / "a.jpg").write_bytes(b"broken")
            self.assertEqual(update()["failed"], 1)
            self.assertEqual(app.read_index(index)[0], {})
            (library / "a.jpg").write_bytes(b"fixed")
            self.assertEqual(update()["added"], 1)
            (library / "a.jpg").unlink()
            self.assertEqual(update()["deleted"], 1)
            self.assertEqual(app.read_index(index)[0], {})
            with self.assertRaises(RuntimeError):
                app.update_index(root / "missing", index, encode)

    def test_failed_save_preserves_previous_index(self):
        with tempfile.TemporaryDirectory() as folder:
            index = Path(folder) / "index.npz"
            records = {"a.jpg": {"sha256": "abc", "status": "ready"}}
            vectors = {"a.jpg": np.ones(512, dtype=np.float32)}
            app.save_index(index, records, vectors)
            before = index.read_bytes()
            with patch.object(app.os, "replace", side_effect=OSError("simulated write failure")):
                with self.assertRaises(OSError):
                    app.save_index(index, {}, {})
            self.assertEqual(before, index.read_bytes())
            self.assertEqual(app.read_index(index)[0], records)


if __name__ == "__main__":
    unittest.main()
