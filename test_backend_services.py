import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from metadata_store import MetadataStore
from vector_index import persist_vector_index, rank_persistent, rank_vectors, threshold_pairs


class BackendServicesTests(unittest.TestCase):
    def test_vector_ranking_is_normalized_and_sorted(self):
        records = {"a.jpg": {}, "b.jpg": {}, "c.jpg": {}}
        vectors = {
            "a.jpg": np.array([2.0, 0.0], dtype=np.float32),
            "b.jpg": np.array([1.0, 1.0], dtype=np.float32),
            "c.jpg": np.array([-1.0, 0.0], dtype=np.float32),
        }
        rows, backend = rank_vectors(records, vectors, np.array([1.0, 0.0], np.float32))
        self.assertEqual([row[0] for row in rows], ["a.jpg", "b.jpg", "c.jpg"])
        self.assertIn(backend, {"numpy-dot-product", "faiss-transient-flat-ip"})

    def test_persistent_index_keeps_ids_and_applies_add_change_delete(self):
        with TemporaryDirectory() as directory:
            index_file = Path(directory) / "images.npz"
            records = {
                "a.jpg": {"sha256": "a1"},
                "b.jpg": {"sha256": "b1"},
            }
            vectors = {
                "a.jpg": np.array([1.0, 0.0], np.float32),
                "b.jpg": np.array([0.0, 1.0], np.float32),
            }
            backend = persist_vector_index(index_file, records, vectors)
            rows, searched_by = rank_persistent(
                index_file, records, vectors, np.array([1.0, 0.0], np.float32)
            )
            self.assertEqual(rows[0][0], "a.jpg")
            self.assertIn(searched_by, {"numpy-dot-product", "faiss-persistent-idmap2"})
            if backend == "faiss-persistent-idmap2":
                import json
                manifest_path = index_file.with_suffix(".faiss.json")
                first = json.loads(manifest_path.read_text(encoding="utf-8"))
                stable_id = first["items"]["b.jpg"]["id"]
                records = {
                    "b.jpg": {"sha256": "b1"},
                    "c.jpg": {"sha256": "c1"},
                }
                vectors = {
                    "b.jpg": np.array([0.0, 1.0], np.float32),
                    "c.jpg": np.array([1.0, 0.0], np.float32),
                }
                persist_vector_index(index_file, records, vectors)
                second = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertNotIn("a.jpg", second["items"])
                self.assertEqual(second["items"]["b.jpg"]["id"], stable_id)
                self.assertIn("c.jpg", second["items"])

    def test_threshold_pairs_returns_only_unique_candidates(self):
        names = ["a.jpg", "b.jpg", "c.jpg"]
        vectors = {
            "a.jpg": np.array([1.0, 0.0], dtype=np.float32),
            "b.jpg": np.array([0.99, 0.01], dtype=np.float32),
            "c.jpg": np.array([0.0, 1.0], dtype=np.float32),
        }
        pairs, backend = threshold_pairs(names, vectors, 0.95)
        self.assertEqual([(left, right) for left, right, _ in pairs], [(0, 1)])
        self.assertGreaterEqual(pairs[0][2], 0.95)
        self.assertIn(backend, {"faiss-range-search", "numpy-block-scan"})

    def test_sqlite_ids_survive_updates_and_deletes_are_tombstoned(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "library"
            library.mkdir()
            image = library / "sample.png"
            Image.new("RGB", (12, 8), "red").save(image)
            store = MetadataStore(root / "index" / "library.sqlite3")
            store.sync(library, {"sample.png": {"sha256": "first"}})
            first = store.list()["items"][0]
            store.sync(library, {"sample.png": {"sha256": "second"}})
            second = store.list()["items"][0]
            self.assertEqual(first["id"], second["id"])
            self.assertEqual(second["sha256"], "second")
            image.unlink()
            store.sync(library, {})
            self.assertEqual(store.list()["total"], 0)
            self.assertEqual(store.stats()["counts"]["deleted"], 1)
