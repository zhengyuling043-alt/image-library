import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from incremental_search import fingerprint, save_index
from group_duplicates import complete_groups, generate_groups, classify_pair, hash_distance, inspect_image
from runtime_compat import popcount


class GroupingTests(unittest.TestCase):
    def test_multiview_hash_is_robust_to_center_crop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = Image.new("RGB", (200, 140), "white")
            for x in range(20, 180, 20):
                for y in range(20, 120, 20):
                    original.paste((x, y, (x + y) % 255), (x, y, x + 12, y + 12))
            original_path = root / "original.png"
            crop_path = root / "crop.png"
            original.save(original_path)
            original.crop((20, 14, 180, 126)).resize((200, 140)).save(crop_path)
            left, right = inspect_image(original_path), inspect_image(crop_path)
            plain = popcount(left["dhash"] ^ right["dhash"])
            robust = hash_distance(left, right)
            self.assertLessEqual(robust, plain)
            self.assertIn("dhashes", left)
    def test_chain_does_not_merge(self):
        edges = {frozenset(("a", "b")), frozenset(("b", "c"))}
        groups = complete_groups(["c", "b", "a"], lambda a, b: frozenset((a, b)) in edges)
        self.assertEqual(groups, [["a", "b"], ["c"]])

    def test_quality_score_rewards_detail_and_exposes_components(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "detail.png"
            pixels = np.indices((64, 64)).sum(axis=0) % 2 * 255
            Image.fromarray(pixels.astype(np.uint8)).save(path)
            info = inspect_image(path)
            self.assertGreater(info["sharpness"], 0)
            self.assertGreater(info["quality_score"], 0)
            self.assertEqual(set(info["quality_components"]),
                             {"resolution", "sharpness", "exposure"})

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
            self.assertIn(group["representative"], group["members"])
            self.assertEqual(set(group["quality_scores"]), set(group["members"]))
            self.assertIn("综合质量分", group["keep_reason"])
            self.assertEqual(sum(p["category"] == "exact" for p in group["pairs"]), 1)
            self.assertEqual(len(list(library.iterdir())), 3)
            (library / "small.png").write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                generate_groups(library, index)


if __name__ == "__main__":
    unittest.main()
