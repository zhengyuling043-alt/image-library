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
