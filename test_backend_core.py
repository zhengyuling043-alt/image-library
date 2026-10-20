import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from backend_core.model_service import encode_image_path, select_device
from backend_core.recycle_service import RecycleBinService, ReviewSigner


class ModelServiceTests(unittest.TestCase):
    def test_device_policy_is_explicit_and_validated(self):
        self.assertEqual(select_device("auto", cuda_available=False), "cpu")
        self.assertEqual(select_device("auto", cuda_available=True), "cuda")
        self.assertEqual(select_device("cpu", cuda_available=True), "cpu")
        with self.assertRaises(RuntimeError):
            select_device("cuda", cuda_available=False)
        with self.assertRaises(RuntimeError):
            select_device("unknown", cuda_available=False)

    def test_image_encoding_returns_normalized_float32_vector(self):
        import torch

        class FakeModel:
            def parameters(self):
                return iter([torch.nn.Parameter(torch.zeros(1))])

            def encode_image(self, tensor):
                self.seen_shape = tuple(tensor.shape)
                return torch.tensor([[3.0, 4.0]], device=tensor.device)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query.png"
            Image.new("RGB", (8, 6), (10, 20, 30)).save(path)
            model = FakeModel()
            vector = encode_image_path(
                path,
                model,
                lambda image: torch.zeros((3, image.height, image.width)),
            )

        self.assertEqual(model.seen_shape, (1, 3, 6, 8))
        self.assertEqual(vector.dtype, np.float32)
        np.testing.assert_allclose(vector, np.array([0.6, 0.8], np.float32))
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=6)


class RecycleServiceTests(unittest.TestCase):
    def test_signer_rejects_modified_review(self):
        signer = ReviewSigner("personal-secret")
        review = {"hashes": {"a.jpg": "abc"}, "thresholds": {"cosine": 0.9}}
        signature = signer.sign(review)
        self.assertTrue(signer.verify(review, signature))
        review["thresholds"]["cosine"] = 0.1
        self.assertFalse(signer.verify(review, signature))

    def test_cleanup_and_restore_are_recoverable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "library"
            library.mkdir()
            index = root / "index" / "images.npz"
            index.parent.mkdir()
            files = {"remove.jpg": b"duplicate-a", "keep.jpg": b"duplicate-b"}
            for name, data in files.items():
                (library / name).write_bytes(data)
            records = {
                name: {"sha256": hashlib.sha256(data).hexdigest()}
                for name, data in files.items()
            }
            vectors = {
                "remove.jpg": np.array([1.0, 0.0], np.float32),
                "keep.jpg": np.array([1.0, 0.0], np.float32),
            }
            storage = {str(index): (records, vectors)}

            def read_index(path):
                saved_records, saved_vectors = storage[str(path)]
                return dict(saved_records), dict(saved_vectors)

            def save_index(path, next_records, next_vectors):
                storage[str(path)] = (dict(next_records), dict(next_vectors))

            service = RecycleBinService(
                library,
                index,
                "personal-secret",
                read_index,
                save_index,
                clock=lambda: 123.0,
                id_factory=lambda: "a" * 32,
            )
            review = {"hashes": {name: item["sha256"] for name, item in records.items()}}
            body = {
                "review": review,
                "signature": service.sign_review(review),
                "names": ["remove.jpg"],
                "keep": "keep.jpg",
            }

            cleaned = service.cleanup(body)
            self.assertEqual(cleaned["deleted"], ["remove.jpg"])
            self.assertFalse((library / "remove.jpg").exists())
            self.assertTrue((library / "keep.jpg").is_file())
            self.assertEqual(service.recoverable_batches(), [
                {"id": "a" * 32, "count": 1, "created": 123.0}
            ])

            restored = service.restore({"batch": "a" * 32})
            self.assertEqual(restored, {"restored": ["remove.jpg"]})
            self.assertEqual((library / "remove.jpg").read_bytes(), files["remove.jpg"])
            self.assertEqual(service.recoverable_batches(), [])


if __name__ == "__main__":
    unittest.main()

