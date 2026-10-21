"""算法评测工具测试（2 条）。

测什么：evaluate_duplicates.py 的指标统计与数据划分——
  test_metrics_count_false_positives_and_misses   查准率/召回率/F1 的四格计数
                                                  （tp/fp/fn/tn）算得对，空数据不误报；
  test_generation_split_integrity_and_no_overwrite 生成评测集时调参/测试分区完整、
                                                  不重叠，且拒绝覆盖已有标注文件。

运行方式：python -m unittest test_evaluate_duplicates -v
"""
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from evaluate_duplicates import (
    cluster_measure, complete_link_cluster_measure, measure, prepare, validate_dataset,
)


class EvaluationTests(unittest.TestCase):
    def test_metrics_count_false_positives_and_misses(self):
        def row(positive, cosine, distance, exact=False):
            return dict(positive=positive, cosine=cosine, distance=distance, exact=exact)
        rows = [row(True, 1, 0, True), row(True, .5, 20),
                row(False, .99, 0), row(False, .5, 20)]
        result = measure(rows, .94, 6)
        self.assertEqual([result[k] for k in ("tp", "fp", "fn", "tn")], [1, 1, 1, 1])
        self.assertEqual(result["precision"], .5)
        self.assertEqual(result["recall"], .5)
        self.assertEqual(result["f1"], .5)
        self.assertEqual(len(result["errors"]), 2)
        self.assertEqual(measure([], .94, 6)["precision"], 0)

    def test_generation_split_integrity_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            source.mkdir()
            for number in range(4):
                Image.new("RGB", (40, 30), (number * 50, 100, 200)).save(source / f"pexels-{number}.png")
            target = root / "evaluation"
            manifest = prepare(source, target, count=4)
            self.assertEqual(len(manifest["samples"]), 28)
            validate_dataset(target, manifest)
            for source_info in manifest["sources"]:
                samples = [s for s in manifest["samples"] if s["source_id"] == source_info["id"]]
                self.assertEqual(len({s["split"] for s in samples}), 1)
                hashes = {s["variant"]: s["sha256"] for s in samples}
                self.assertEqual(hashes["original"], hashes["copy"])
            with self.assertRaises(RuntimeError):
                prepare(source, target, count=4)
            sample = manifest["samples"][0]
            (target / sample["split"] / "images" / sample["file"]).write_bytes(b"edited")
            with self.assertRaises(RuntimeError):
                validate_dataset(target, manifest)

    def test_cluster_metrics_capture_transitive_false_merge(self):
        rows = [
            {"a": "a", "b": "b", "positive": True, "exact": False, "cosine": .99, "distance": 1},
            {"a": "a", "b": "c", "positive": False, "exact": False, "cosine": .20, "distance": 40},
            {"a": "b", "b": "c", "positive": False, "exact": False, "cosine": .99, "distance": 1},
        ]
        result = cluster_measure(rows, .94, 6)
        self.assertLess(result["precision"], 1.0)
        self.assertEqual(result["recall"], 1.0)
        self.assertLess(result["purity"], 1.0)

        protected = complete_link_cluster_measure(rows, .94, 6)
        self.assertEqual(protected["precision"], 1.0)
        self.assertEqual(protected["recall"], 1.0)
        self.assertEqual(protected["purity"], 1.0)
        self.assertEqual(protected["overmerge_groups"], 0)


if __name__ == "__main__":
    unittest.main()
