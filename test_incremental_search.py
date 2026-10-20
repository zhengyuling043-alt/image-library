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

    def test_interrupted_batch_resumes_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            library, index = root / "library", root / "index.npz"
            library.mkdir()
            for number in range(30):
                (library / f"{number:02d}.jpg").write_bytes(str(number).encode())
            calls = []

            def interrupted(path):
                calls.append(path.name)
                if len(calls) == 26:
                    raise KeyboardInterrupt("simulated stop")
                vector = np.zeros(512, np.float32)
                vector[int(path.stem) % 512] = 1
                return vector

            with self.assertRaises(KeyboardInterrupt):
                app.update_index(library, index, interrupted, checkpoint_every=25)
            self.assertTrue(index.with_suffix(".checkpoint.json").is_file())
            self.assertEqual(len(app.read_index(index)[0]), 25)

            resumed_calls = []
            def resumed_encoder(path):
                resumed_calls.append(path.name)
                vector = np.zeros(512, np.float32)
                vector[int(path.stem) % 512] = 1
                return vector

            report = app.update_index(library, index, resumed_encoder, checkpoint_every=25)
            self.assertTrue(report["resumed"])
            self.assertEqual(report["ready"], 30)
            self.assertEqual(resumed_calls, [f"{number:02d}.jpg" for number in range(25, 30)])
            self.assertFalse(index.with_suffix(".checkpoint.json").exists())

    def test_bulk_encoder_is_used_for_changed_images_only(self):
        class BulkEncoder:
            def __init__(self):
                self.calls = []

            def encode_many(self, paths, batch_size=None):
                paths = list(paths)
                self.calls.append([path.name for path in paths])
                return {
                    path: np.eye(1, 512, int(path.stem), dtype=np.float32)[0]
                    for path in paths
                }, {}

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            library, index = root / "library", root / "index.npz"
            library.mkdir()
            (library / "0.jpg").write_bytes(b"zero")
            encoder = BulkEncoder()
            first = app.update_index(library, index, encoder)
            self.assertEqual(first["counts"]["added"], 1)
            self.assertEqual(encoder.calls, [["0.jpg"]])
            encoder.calls.clear()
            (library / "1.jpg").write_bytes(b"one")
            second = app.update_index(library, index, encoder)
            self.assertEqual(second["counts"]["reused"], 1)
            self.assertEqual(second["counts"]["added"], 1)
            self.assertEqual(encoder.calls, [["1.jpg"]])


if __name__ == "__main__":
    unittest.main()
