import tempfile
import unittest
from pathlib import Path

from PIL import Image

from evaluate_duplicates import prepare, validate_dataset, measure


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


if __name__ == "__main__":
    unittest.main()
