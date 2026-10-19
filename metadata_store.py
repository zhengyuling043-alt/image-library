"""SQLite metadata catalogue with stable image identifiers."""

import sqlite3
import time
from pathlib import Path

from PIL import Image, ImageOps


SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    sha256 TEXT,
    status TEXT NOT NULL,
    width INTEGER NOT NULL DEFAULT 0,
    height INTEGER NOT NULL DEFAULT 0,
    bytes INTEGER NOT NULL DEFAULT 0,
    format TEXT,
    mtime_ns INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    updated_at REAL NOT NULL,
    deleted_at REAL
);
CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);
CREATE INDEX IF NOT EXISTS idx_images_sha256 ON images(sha256);
"""


class MetadataStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def _connect(self):
        connection = sqlite3.connect(str(self.path), timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def sync(self, library, records, failures=None):
        """Synchronise filesystem/index state while preserving numeric IDs."""

        library, failures = Path(library), failures or {}
        now, present = time.time(), set()
        with self._connect() as connection:
            for path in sorted(library.iterdir()):
                if not path.is_file():
                    continue
                present.add(path.name)
                stat, width, height, image_format, error = path.stat(), 0, 0, None, failures.get(path.name)
                try:
                    with Image.open(path) as source:
                        image = ImageOps.exif_transpose(source)
                        width, height = image.size
                        image_format = source.format
                except (OSError, ValueError, Image.DecompressionBombError) as exc:
                    error = error or str(exc)
                record = records.get(path.name, {})
                status = "ready" if record else ("failed" if error else "pending")
                connection.execute(
                    """INSERT INTO images(name,sha256,status,width,height,bytes,format,mtime_ns,error,updated_at,deleted_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,NULL)
                       ON CONFLICT(name) DO UPDATE SET sha256=excluded.sha256,status=excluded.status,
                       width=excluded.width,height=excluded.height,bytes=excluded.bytes,format=excluded.format,
                       mtime_ns=excluded.mtime_ns,error=excluded.error,updated_at=excluded.updated_at,deleted_at=NULL""",
                    (path.name, record.get("sha256"), status, width, height, stat.st_size,
                     image_format, stat.st_mtime_ns, error, now),
                )
            rows = connection.execute("SELECT name FROM images WHERE deleted_at IS NULL").fetchall()
            for row in rows:
                if row["name"] not in present:
                    connection.execute(
                        "UPDATE images SET status='deleted', deleted_at=?, updated_at=? WHERE name=?",
                        (now, now, row["name"]),
                    )

    def stats(self):
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM images GROUP BY status"
            ).fetchall()
        counts = {row["status"]: row["count"] for row in rows}
        return {"path": str(self.path.resolve()), "counts": counts,
                "active": sum(value for key, value in counts.items() if key != "deleted")}

    def list(self, limit=50, offset=0, status=None, query=None):
        limit, offset = max(1, min(int(limit), 200)), max(0, int(offset))
        clauses, values = ["deleted_at IS NULL"], []
        if status:
            clauses.append("status=?")
            values.append(status)
        if query:
            clauses.append("name LIKE ? ESCAPE '\\'")
            escaped = str(query).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            values.append("%" + escaped + "%")
        where = " AND ".join(clauses)
        with self._connect() as connection:
            total = connection.execute("SELECT COUNT(*) FROM images WHERE " + where, values).fetchone()[0]
            rows = connection.execute(
                "SELECT * FROM images WHERE " + where + " ORDER BY name LIMIT ? OFFSET ?",
                [*values, limit, offset],
            ).fetchall()
        return {"items": [dict(row) for row in rows], "total": total,
                "limit": limit, "offset": offset}
