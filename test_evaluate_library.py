import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_library import evaluate
from incremental_search import fingerprint, save_index


class ExistingLibraryEvaluationTests(unittest.TestCase):
    def test_manifest_split_evaluation_is_reproducible(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            library = root / "library"
            index = root / "index" / "incremental.npz"
            manifest_file = root / "manifest.json"
            json_report = root / "result.json"
            markdown_report = root / "result.md"
            library.mkdir()
            files, groups, records, vectors = [], [], {}, {}
            for source_number in range(4):
                source_id = f"source_{source_number}"
                members = []
                original = library / f"{source_id}_original.png"
                Image.new("RGB", (32, 24), (source_number * 50, 80, 160)).save(original)
                copy = library / f"{source_id}_copy.png"
                copy.write_bytes(original.read_bytes())
                vector = np.zeros(512, dtype=np.float32)
                vector[source_number] = 1.0
                for path, transform in ((original, "original"), (copy, "copy")):
                    members.append(path.name)
                    files.append({"name": path.name, "source_id": source_id,
                                  "transform": transform})
                    records[path.name] = {"sha256": fingerprint(path)}
                    vectors[path.name] = vector
                groups.append({"source_id": source_id, "members": members,
                               "split": "calibration" if source_number < 2 else "test"})
            save_index(index, records, vectors)
            manifest_file.write_text(json.dumps({
                "files": files, "groups": groups,
                "note": "synthetic source groups",
            }), encoding="utf-8")

            result = evaluate(library, index, manifest_file, json_report, markdown_report)

            self.assertEqual(result["integrity"]["manifest_images"], 8)
            self.assertEqual(result["splits"]["test"]["sources"], 2)
            self.assertEqual(result["test_selected"]["precision"], 1.0)
            self.assertEqual(result["test_selected"]["recall"], 1.0)
            self.assertEqual(result["test_cluster_complete_link"]["overmerge_groups"], 0)
            self.assertTrue(json_report.is_file())
            self.assertIn("传递性处理对比", markdown_report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
