"""Storage: one SQLite table for the records, and JPEG files for the photos.

Photos are kept as plain files under IMAGE_DIR (one folder per day) and the
table stores their path relative to that folder. Files are easier to look at,
back up and serve than images stored inside the database.
"""

import secrets
import sqlite3
import threading
from pathlib import Path

import cv2

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS sightings (
    id            INTEGER PRIMARY KEY,
    time          TEXT NOT NULL,     -- when the plate was first seen
    source        TEXT NOT NULL,     -- video file or stream
    plate         TEXT NOT NULL,
    confidence    INTEGER NOT NULL,  -- 0-100
    readings      INTEGER NOT NULL,  -- OCR readings that agreed
    direction     TEXT,              -- 'in', 'out' or NULL
    vehicle_type  TEXT,              -- car, motorcycle, bus, truck or NULL
    colour        TEXT,
    make          TEXT,              -- NULL when the model was not sure
    plate_image   TEXT,              -- paths relative to the image folder
    vehicle_image TEXT
);
CREATE INDEX IF NOT EXISTS sightings_time ON sightings(time);
CREATE INDEX IF NOT EXISTS sightings_plate ON sightings(plate);
"""

FIELDS = ("time", "source", "plate", "confidence", "readings", "direction",
          "vehicle_type", "colour", "make", "plate_image", "vehicle_image")


class Database:
    """Thread-safe: the recorder writes while the dashboard reads."""

    def __init__(self, path=config.DB_FILE, image_dir=config.IMAGE_DIR):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.image_dir = Path(image_dir)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)

    # ---------- photos ----------
    def save_image(self, img, day, name):
        """Write a JPEG and return its path relative to the image folder."""
        rel = Path(day) / f"{name}_{secrets.token_hex(3)}.jpg"
        (self.image_dir / day).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.image_dir / rel), img,
                    [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_SAVE_QUALITY])
        return rel.as_posix()

    def delete_images(self, *rels):
        for rel in rels:
            if rel:
                (self.image_dir / rel).unlink(missing_ok=True)

    # ---------- records ----------
    def insert(self, row):
        with self._lock, self._db:
            cur = self._db.execute(
                f"INSERT INTO sightings ({', '.join(FIELDS)}) "
                f"VALUES ({', '.join('?' * len(FIELDS))})",
                [row.get(k) for k in FIELDS])
            return cur.lastrowid

    def update(self, row_id, row):
        keys = [k for k in FIELDS if k in row]
        with self._lock, self._db:
            self._db.execute(
                f"UPDATE sightings SET {', '.join(f'{k} = ?' for k in keys)} WHERE id = ?",
                [row[k] for k in keys] + [row_id])

    def get(self, row_id):
        with self._lock:
            r = self._db.execute("SELECT * FROM sightings WHERE id = ?", (row_id,)).fetchone()
        return dict(r) if r else None

    def recent(self, limit=200):
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM sightings ORDER BY time DESC, id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def stats(self):
        with self._lock:
            r = self._db.execute(
                "SELECT COUNT(*) AS total, COUNT(DISTINCT plate) AS unique_plates, "
                "SUM(direction = 'in') AS incoming, SUM(direction = 'out') AS outgoing "
                "FROM sightings").fetchone()
        return {k: r[k] or 0 for k in r.keys()}

    def close(self):
        with self._lock:
            self._db.close()
