"""Signed duplicate cleanup and recoverable soft-delete service."""

import hashlib
import hmac
import json
import re
import time
import uuid
from pathlib import Path


def _safe_file(folder, name):
    if (not isinstance(name, str) or not name or "/" in name or "\\" in name
            or ":" in name or name in {".", ".."}):
        raise ValueError("无效文件名")
    folder = Path(folder).resolve()
    path = (folder / name).resolve()
    if path.parent != folder:
        raise ValueError("文件路径超出允许目录")
    return path


def _fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ReviewSigner:
    """Create and verify canonical HMAC-SHA256 review snapshots."""

    def __init__(self, secret):
        if not isinstance(secret, str) or not secret:
            raise ValueError("签名密钥不能为空")
        self._secret = secret.encode("utf-8")

    @staticmethod
    def canonical(review):
        if not isinstance(review, dict):
            raise ValueError("分组审查数据无效")
        return json.dumps(review, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":")).encode("utf-8")

    def sign(self, review):
        return hmac.new(self._secret, self.canonical(review), hashlib.sha256).hexdigest()

    def verify(self, review, signature):
        return isinstance(signature, str) and hmac.compare_digest(
            signature, self.sign(review)
        )


class RecycleBinService:
    """Move duplicate versions to signed batches and restore them transactionally."""

    def __init__(self, library, index_file, secret, read_index, save_index,
                 metadata=None, clock=time.time, id_factory=None):
        self.library = Path(library)
        self.index_file = Path(index_file)
        self.trash = self.index_file.parent / "trash"
        self.signer = ReviewSigner(secret)
        self.read_index = read_index
        self.save_index = save_index
        self.metadata = metadata
        self.clock = clock
        self.id_factory = id_factory or (lambda: uuid.uuid4().hex)

    def sign_review(self, review):
        return self.signer.sign(review)

    def recoverable_batches(self):
        batches = []
        if not self.trash.exists():
            return batches
        for path in self.trash.glob("*/receipt.json"):
            try:
                receipt = json.loads(path.read_text(encoding="utf-8"))
                remaining = [
                    name for name in receipt["files"]
                    if _safe_file(path.parent / "files", name).is_file()
                ]
                if remaining:
                    batches.append({
                        "id": path.parent.name,
                        "count": len(remaining),
                        "created": receipt["created"],
                    })
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
        return sorted(batches, key=lambda item: item["created"], reverse=True)

    def cleanup(self, body):
        review = body.get("review")
        signature = body.get("signature", "")
        if not isinstance(review, dict) or not self.signer.verify(review, signature):
            raise ValueError("分组结果已失效，请重新分析后清理")

        names = body.get("names")
        if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
            raise ValueError("请先选择要清理的重复版本")
        names = list(dict.fromkeys(names))
        members = review.get("hashes")
        keep = body.get("keep")
        if (not isinstance(members, dict) or keep not in members or keep in names
                or any(name not in members for name in names)):
            raise ValueError("清理范围不属于已查看的重复组，或包含要保留的图片")

        records, vectors = self.read_index(self.index_file)
        for name, digest in members.items():
            path = _safe_file(self.library, name)
            if (not path.is_file() or _fingerprint(path) != digest
                    or records.get(name, {}).get("sha256") != digest):
                raise ValueError("组内图片已变化，请更新索引并重新分析；本次未清理任何图片")

        batch_id = self.id_factory()
        if not re.fullmatch(r"[0-9a-f]{32}", batch_id):
            raise RuntimeError("回收批次生成器返回了无效ID")
        batch = self.trash / batch_id
        storage = batch / "files"
        storage.mkdir(parents=True, exist_ok=False)
        receipt = {
            "created": self.clock(),
            "files": {name: records[name]["sha256"] for name in names},
            "keep": keep,
        }
        (batch / "receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False), encoding="utf-8"
        )
        self.save_index(
            batch / "vectors.npz",
            {name: records[name] for name in names},
            {name: vectors[name] for name in names},
        )

        moved = []
        try:
            for name in names:
                _safe_file(self.library, name).replace(_safe_file(storage, name))
                moved.append(name)
            remaining_records = {name: value for name, value in records.items() if name not in names}
            remaining_vectors = {name: value for name, value in vectors.items() if name not in names}
            self.save_index(self.index_file, remaining_records, remaining_vectors)
            if self.metadata is not None:
                self.metadata.sync(self.library, remaining_records)
        except Exception:
            for name in reversed(moved):
                source = _safe_file(storage, name)
                if source.is_file():
                    source.replace(_safe_file(self.library, name))
            raise

        return {
            "deleted": names,
            "rejected": [],
            "batch": batch_id,
            "counts": {"deleted": len(names)},
            "message": "已移入本地回收区，可恢复",
        }

    def restore(self, body):
        identifier = body.get("batch", "")
        if not isinstance(identifier, str) or not re.fullmatch(r"[0-9a-f]{32}", identifier):
            raise ValueError("无效回收批次")
        batch = self.trash / identifier
        try:
            receipt = json.loads((batch / "receipt.json").read_text(encoding="utf-8"))
            old_records, old_vectors = self.read_index(batch / "vectors.npz")
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise ValueError("回收批次不存在或已损坏") from error

        records, vectors = self.read_index(self.index_file)
        names = [
            name for name in receipt["files"]
            if _safe_file(batch / "files", name).is_file()
        ]
        if not names:
            raise ValueError("该批次已恢复")
        for name in names:
            if _safe_file(self.library, name).exists() or name in records:
                raise ValueError(f"素材库已有同名文件 {name}，为避免覆盖，本次未恢复")
            if _fingerprint(_safe_file(batch / "files", name)) != receipt["files"][name]:
                raise ValueError("回收文件已变化，停止恢复")
            if name not in old_records or name not in old_vectors:
                raise ValueError("回收批次缺少向量数据")

        moved = []
        try:
            for name in names:
                _safe_file(batch / "files", name).replace(_safe_file(self.library, name))
                moved.append(name)
            restored_records = {**records, **{name: old_records[name] for name in names}}
            restored_vectors = {**vectors, **{name: old_vectors[name] for name in names}}
            self.save_index(self.index_file, restored_records, restored_vectors)
            if self.metadata is not None:
                self.metadata.sync(self.library, restored_records)
        except Exception:
            for name in reversed(moved):
                source = _safe_file(self.library, name)
                if source.is_file():
                    source.replace(_safe_file(batch / "files", name))
            raise
        return {"restored": names}

