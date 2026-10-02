import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from incremental_search import fingerprint, save_index
from group_duplicates import complete_groups, generate_groups, classify_pair


class GroupingTests(unittest.TestCase):
    def test_chain_does_not_merge(self):
        edges = {frozenset(("a", "b")), frozenset(("b", "c"))}
        groups = complete_groups(["c", "b", "a"], lambda a, b: frozenset((a, b)) in edges)
        self.assertEqual(groups, [["a", "b"], ["c"]])

    def test_semantic_similarity_alone_is_not_duplicate(self):
        info = {"a": {"sha256": "a", "dhash": 0},
                "b": {"sha256": "b", "dhash": 2**64 - 1}}
        vectors = {n: np.array([1., 0.]) for n in info}
        self.assertEqual(classify_pair("a", "b", info, vectors, .94, 6)["category"], "similar")
        info["b"]["sha256"] = "a"
        self.assertEqual(classify_pair("a", "b", info, vectors, .94, 6)["category"], "exact")

    def test_real_images_report_and_stale_index(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            library = root / "library"
            library.mkdir()
            # 非均匀图案及其缩小版；仅用替代向量测试分组集成。
            pixels = np.tile(np.arange(64, dtype=np.uint8) * 4, (64, 1))
            image = Image.fromarray(pixels)
            image.save(library / "original.png")
            shutil.copyfile(library / "original.png", library / "copy.png")
            image.resize((32, 32)).save(library / "small.png")
            records = {p.name: {"sha256": fingerprint(p)} for p in sorted(library.iterdir())}
            vector = np.zeros(512, dtype=np.float32)
            vector[0] = 1
            index = root / "index" / "incremental.npz"
            save_index(index, records, {n: vector for n in records})
            report = generate_groups(library, index)
            self.assertEqual(len(report["groups"]), 1)
            group = report["groups"][0]
            self.assertEqual(len(group["members"]), 3)
            self.assertEqual(group["category"], "near")
            self.assertNotEqual(group["suggested_keep"], "small.png")
            self.assertEqual(sum(p["category"] == "exact" for p in group["pairs"]), 1)
            self.assertEqual(len(list(library.iterdir())), 3)
            (library / "small.png").write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                generate_groups(library, index)


if __name__ == "__main__":
    unittest.main()
