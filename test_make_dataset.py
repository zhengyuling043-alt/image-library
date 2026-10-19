"""合成数据生成器测试（2 条）。

测什么：make_dataset.py 的生成正确性——
  test_stripes_preserve_requested_dimensions        条纹图片生成严格保持请求的尺寸；
  test_generation_refuses_to_overwrite_existing_labels 目标标注文件已存在时拒绝生成，
                                                  防止意外覆盖人工数据。

运行方式：python -m unittest test_make_dataset -v
"""
import random
import tempfile
import unittest
from pathlib import Path

from make_dataset import stripes, generate


class SyntheticDatasetTests(unittest.TestCase):
    def test_stripes_preserve_requested_dimensions(self):
        for seed in range(20):
            self.assertEqual(stripes(random.Random(seed), 210, 170).size, (210, 170))

    def test_generation_refuses_to_overwrite_existing_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "labels.json"
            labels.write_text("existing", encoding="utf-8")
            with self.assertRaises(ValueError):
                generate(1, 42, root / "images", labels)
            self.assertEqual(labels.read_text(encoding="utf-8"), "existing")
