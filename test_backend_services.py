import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from metadata_store import MetadataStore
from vector_index import rank_vectors, threshold_pairs


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
        self.assertIn(backend, {"numpy-dot-product", "faiss-index-flat-ip"})

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
