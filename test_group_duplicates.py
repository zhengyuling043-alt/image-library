"""重复分组算法测试（9 条）。

测什么：group_duplicates.py 的聚类正确性——
  test_chain_does_not_merge                       传递性阻断：甲≈乙、乙≈丙但甲≉丙时，
                                                  不得强行并成一组（核心难点，组内两两校验策略的验证）；
  test_semantic_similarity_alone_is_not_duplicate 双通道判定：只有 CLIP 语义相似、dHash
                                                  差异大 → 判 similar 不算重复；内容哈希相同 → exact；
  test_real_images_report_and_stale_index         真实图片跑完整分组流程，校验报告结构
                                                  与过期索引检测；
  test_empty_library_groups_into_no_groups        空库边界：无图片、无记录时报告结构与
                                                  候选计数保持稳定，不抛异常；
  test_single_image_stays_singleton               单图边界：唯一图片应落在 singletons，
                                                  不得生成空组；
  test_all_identical_images_merge_into_one_group  全重复边界：多张字节级相同的图片应
                                                  正确归并为单个 exact 组。

运行方式：python -m unittest test_group_duplicates -v
"""
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

    def test_empty_library_groups_into_no_groups(self):
        """空库边界：索引与素材均为空时，报告结构稳定且候选计数为 0。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "library"
            library.mkdir()
            index = root / "index" / "incremental.npz"
            save_index(index, {}, {})
            report = generate_groups(library, index, write_report=False)
            self.assertEqual(report["groups"], [])
            self.assertEqual(report["singletons"], [])
            self.assertEqual(report["skipped"], {})
            self.assertEqual(report["candidate_pair_count"], 0)

    def test_single_image_stays_singleton(self):
        """单图边界：唯一图片应落入 singletons，不产生任何重复组。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "library"
            library.mkdir()
            pixels = np.tile(np.arange(64, dtype=np.uint8) * 4, (64, 1))
            Image.fromarray(pixels).save(library / "only.png")
            records = {p.name: {"sha256": fingerprint(p)} for p in sorted(library.iterdir())}
            vector = np.zeros(512, dtype=np.float32)
            vector[0] = 1
            index = root / "index" / "incremental.npz"
            save_index(index, records, {n: vector for n in records})
            report = generate_groups(library, index, write_report=False)
            self.assertEqual(report["groups"], [])
            self.assertEqual(report["singletons"], ["only.png"])
            self.assertEqual(report["candidate_pair_count"], 0)

    def test_all_identical_images_merge_into_one_group(self):
        """全重复边界：多张字节级相同的图片应归并为单个 exact 组。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "library"
            library.mkdir()
            pixels = np.tile(np.arange(64, dtype=np.uint8) * 4, (64, 1))
            image = Image.fromarray(pixels)
            image.save(library / "a.png")
            shutil.copyfile(library / "a.png", library / "b.png")
            shutil.copyfile(library / "a.png", library / "c.png")
            records = {p.name: {"sha256": fingerprint(p)} for p in sorted(library.iterdir())}
            vector = np.zeros(512, dtype=np.float32)
            vector[0] = 1
            index = root / "index" / "incremental.npz"
            save_index(index, records, {n: vector for n in records})
            report = generate_groups(library, index, write_report=False)
            self.assertEqual(len(report["groups"]), 1)
            group = report["groups"][0]
            self.assertEqual(group["category"], "exact")
            self.assertEqual(set(group["members"]), {"a.png", "b.png", "c.png"})
            self.assertEqual(group["pair_count"], 3)
            self.assertTrue(all(p["category"] == "exact" for p in group["pairs"]))
            self.assertIn(group["suggested_keep"], group["members"])


if __name__ == "__main__":
    unittest.main()
