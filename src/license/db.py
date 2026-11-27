"""Storage: SQLite for the records, jobs, cameras and watchlist; JPEG files for photos.

Photos and videos are kept as plain files under DATA_DIR (images one folder
per day) and the tables store their path relative to that folder. Files are
easier to look at, back up and serve than data stored inside the database.
The data folder is created readable by the current user only.
"""

import csv
import io
import os
import secrets
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path

import cv2

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS sightings (
    id            INTEGER PRIMARY KEY,
    time          TEXT NOT NULL,     -- when the record was made (date and time of processing)
    source        TEXT NOT NULL,     -- video file name or camera
    plate         TEXT NOT NULL DEFAULT '',   -- '' when unreadable
    confidence    INTEGER NOT NULL DEFAULT 0, -- OCR confidence 0-100
    readings      INTEGER NOT NULL DEFAULT 0, -- OCR readings that agreed
    direction     TEXT,              -- 'in', 'out' or NULL
    vehicle_type  TEXT,              -- car, van, motorcycle, bus, truck or NULL
    colour        TEXT,
    make          TEXT,              -- NULL when the model was not sure
    plate_image   TEXT,              -- paths relative to the image folder
    vehicle_image TEXT
);
CREATE TABLE IF NOT EXISTS jobs (
    id        INTEGER PRIMARY KEY,
    kind      TEXT NOT NULL,         -- upload, file or camera
    name      TEXT NOT NULL,         -- file name or camera name
    path      TEXT NOT NULL,         -- video file or stream URL
    status    TEXT NOT NULL,         -- queued, processing, done, failed, stopped
    progress  REAL NOT NULL DEFAULT 0,
    frames    INTEGER NOT NULL DEFAULT 0,
    fps       REAL,                  -- processing speed
    duration  REAL,                  -- video length in seconds
    error     TEXT,
    output    TEXT,                  -- annotated video, relative to the video folder
    created   TEXT NOT NULL,
    started   TEXT,
    finished  TEXT
);
CREATE TABLE IF NOT EXISTS cameras (
    id       INTEGER PRIMARY KEY,
    name     TEXT NOT NULL,
    url      TEXT NOT NULL,          -- RTSP/HTTP URL, or a device number such as 0
    created  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS watchlist (
    id         INTEGER PRIMARY KEY,
    plate      TEXT NOT NULL,
    plate_key  TEXT NOT NULL UNIQUE,
    kind       TEXT NOT NULL,        -- allow (whitelist) or block (blacklist)
    note       TEXT,
    created    TEXT NOT NULL
);
"""

# Columns added after the first version; old databases get them on start-up
NEW_COLUMNS = {
    "video_time": "REAL",                       # seconds from the start of the video
    "job_id": "INTEGER",
    "track_id": "INTEGER",                      # vehicle tracking ID (ByteTrack)
    "plate_key": "TEXT NOT NULL DEFAULT ''",    # plate normalised for matching
    "status": "TEXT NOT NULL DEFAULT 'recognized'",   # recognized, review, unrecognized
    "reviewed": "INTEGER NOT NULL DEFAULT 0",   # 1 once a person checked it
}
INDEXES = """
CREATE INDEX IF NOT EXISTS sightings_time ON sightings(time);
CREATE INDEX IF NOT EXISTS sightings_plate ON sightings(plate);
CREATE INDEX IF NOT EXISTS sightings_key ON sightings(plate_key);
CREATE INDEX IF NOT EXISTS sightings_job ON sightings(job_id);
"""

FIELDS = ("time", "video_time", "source", "job_id", "track_id", "plate", "plate_key",
          "confidence", "readings", "status", "direction", "vehicle_type", "colour", "make",
          "plate_image", "vehicle_image", "reviewed")
STATUSES = ("recognized", "review", "unrecognized")
TYPES = ("car", "van", "motorcycle", "bus", "truck")
CONFUSED = str.maketrans("OIZSB", "01258")      # characters OCR easily mixes up


def plate_key(text):
    """Plate text for matching: upper case, letters and digits only, and
    look-alike characters treated as equal (O=0, I=1, ...)."""
    return "".join(c for c in (text or "").upper() if c.isalnum()).translate(CONFUSED)


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _private_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    return path


class Database:
    """Thread-safe: the recorder and jobs write while the dashboard reads."""

    def __init__(self, path=config.DB_FILE):
        _private_dir(Path(path).parent)
        self.image_dir = _private_dir(config.IMAGE_DIR)
        self.video_dir = _private_dir(config.VIDEO_DIR)
        self.upload_dir = _private_dir(config.UPLOAD_DIR)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        self._migrate()
        self._db.executescript(INDEXES)

    def _migrate(self):
        have = {r["name"] for r in self._db.execute("PRAGMA table_info(sightings)")}
        for name, kind in NEW_COLUMNS.items():
            if name not in have:
                self._db.execute(f"ALTER TABLE sightings ADD COLUMN {name} {kind}")
        rows = self._db.execute("SELECT id, plate FROM sightings WHERE plate_key = '' AND plate != ''")
        self._db.executemany("UPDATE sightings SET plate_key = ? WHERE id = ?",
                             [(plate_key(r["plate"]), r["id"]) for r in rows.fetchall()])
        self._db.commit()

    def _q(self, sql, args=()):
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, args).fetchall()]

    def _x(self, sql, args=()):
        with self._lock, self._db:
            return self._db.execute(sql, args)

    # ---------- photos ----------
    def save_image(self, img, day, name):
        """Write a JPEG and return its path relative to the image folder."""
        rel = Path(day) / f"{name}_{secrets.token_hex(3)}.jpg"
        _private_dir(self.image_dir / day)
        cv2.imwrite(str(self.image_dir / rel), img,
                    [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_SAVE_QUALITY])
        return rel.as_posix()

    def delete_images(self, *rels):
        for rel in rels:
            if rel:
                (self.image_dir / rel).unlink(missing_ok=True)

    # ---------- records ----------
    def insert(self, row):
        row = {**row, "plate_key": plate_key(row.get("plate"))}
        keys = [k for k in FIELDS if row.get(k) is not None]     # the rest take their defaults
        return self._x(f"INSERT INTO sightings ({', '.join(keys)}) "
                       f"VALUES ({', '.join('?' * len(keys))})",
                       [row[k] for k in keys]).lastrowid

    def update(self, row_id, row):
        if "plate" in row:
            row = {**row, "plate_key": plate_key(row["plate"])}
        keys = [k for k in FIELDS if k in row]
        if keys:
            self._x(f"UPDATE sightings SET {', '.join(f'{k} = ?' for k in keys)} WHERE id = ?",
                    [row[k] for k in keys] + [row_id])

    def delete(self, row_id):
        r = self.get(row_id)
        if r:
            self._x("DELETE FROM sightings WHERE id = ?", (row_id,))
            self.delete_images(r["plate_image"], r["vehicle_image"])
        return r is not None

    SELECT = ("SELECT s.*, w.kind AS watch, w.note AS watch_note FROM sightings s "
              "LEFT JOIN watchlist w ON w.plate_key = s.plate_key AND s.plate_key != ''")

    def get(self, row_id):
        rows = self._q(self.SELECT + " WHERE s.id = ?", (row_id,))
        return rows[0] if rows else None

    @staticmethod
    def _where(f):
        """SQL conditions for the record filters (all optional):
        q      words that must each match: part of the plate, time or source,
               the start of the make, or exactly the colour, type, direction or status
        type, status, direction, watch ('allow'/'block'), job, source
        since, until   date/time range on the record time ('YYYY-MM-DD[ HH:MM]')"""
        where, args = [], []
        for word in (f.get("q") or "").split():
            w = word.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            where.append("(s.plate LIKE ? ESCAPE '\\' OR s.plate_key LIKE ? ESCAPE '\\' "
                         "OR s.time LIKE ? ESCAPE '\\' OR s.source LIKE ? ESCAPE '\\' "
                         "OR s.make LIKE ? ESCAPE '\\' OR s.colour = ? COLLATE NOCASE "
                         "OR s.vehicle_type = ? COLLATE NOCASE OR s.direction = ? COLLATE NOCASE "
                         "OR s.status = ? COLLATE NOCASE)")
            args += [f"%{w}%", f"%{plate_key(word)}%", f"%{w}%", f"%{w}%", f"{w}%",
                     word, word, word, word]
        for key, col in (("type", "s.vehicle_type"), ("status", "s.status"),
                         ("direction", "s.direction"), ("watch", "w.kind"),
                         ("job", "s.job_id"), ("source", "s.source")):
            if f.get(key):
                if f[key] == "none":
                    where.append(f"{col} IS NULL")
                else:
                    where.append(f"{col} = ?")
                    args.append(f[key])
        if f.get("since"):
            where.append("s.time >= ?")
            args.append(f["since"].replace("T", " "))
        if f.get("until"):
            until = f["until"].replace("T", " ")
            where.append("s.time <= ?")
            args.append(until + ("" if len(until) > 16 else ":59" if len(until) > 10 else " 23:59:59"))
        return (" WHERE " + " AND ".join(where)) if where else "", args

    def records(self, filters=None, limit=50, offset=0):
        """Filtered records and the total number that match."""
        f = filters or {}
        where, args = self._where(f)
        order = "ASC" if f.get("sort") == "oldest" else "DESC"
        rows = self._q(f"{self.SELECT}{where} ORDER BY s.time {order}, s.id {order} "
                       f"LIMIT ? OFFSET ?", args + [limit, offset])
        total = self._q(f"SELECT COUNT(*) AS n FROM sightings s LEFT JOIN watchlist w "
                        f"ON w.plate_key = s.plate_key AND s.plate_key != ''{where}", args)[0]["n"]
        return rows, total

    def recent(self, limit=200, query=""):
        return self.records({"q": query}, limit)[0]

    def latest_id(self):
        return self._q("SELECT COALESCE(MAX(id), 0) AS n FROM sightings")[0]["n"]

    def stats(self, filters=None):
        where, args = self._where(filters or {})
        base = (f"FROM sightings s LEFT JOIN watchlist w ON w.plate_key = s.plate_key "
                f"AND s.plate_key != ''{where}")
        r = self._q(
            "SELECT COUNT(*) AS total, COUNT(DISTINCT NULLIF(s.plate_key, '')) AS unique_plates, "
            "SUM(s.status = 'recognized') AS recognized, SUM(s.status = 'review') AS review, "
            "SUM(s.status = 'unrecognized') AS unrecognized, "
            "SUM(s.direction = 'in') AS incoming, SUM(s.direction = 'out') AS outgoing, "
            "SUM(w.kind = 'block') AS blocked, SUM(w.kind = 'allow') AS allowed " + base, args)[0]
        out = {k: v or 0 for k, v in r.items()}
        out["types"] = {t["type"] or "unknown": t["n"] for t in self._q(
            f"SELECT s.vehicle_type AS type, COUNT(*) AS n {base} GROUP BY s.vehicle_type "
            f"ORDER BY n DESC", args)}
        return out

    def trend(self, filters=None):
        """Records over time, per vehicle type. The bucket size follows the
        time span: 5 minutes, 1 hour or 1 day."""
        where, args = self._where(filters or {})
        span = self._q(f"SELECT MIN(s.time) AS a, MAX(s.time) AS b FROM sightings s "
                       f"LEFT JOIN watchlist w ON w.plate_key = s.plate_key "
                       f"AND s.plate_key != ''{where}", args)[0]
        if not span["a"]:
            return {"bucket": "hour", "points": []}
        a, b = (datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in (span["a"], span["b"]))
        if b - a <= timedelta(hours=3):
            bucket, step, fmt = "5min", timedelta(minutes=5), "%Y-%m-%d %H:%M"
            key = ("strftime('%Y-%m-%d %H:', s.time) || "
                   "printf('%02d', (CAST(strftime('%M', s.time) AS INTEGER) / 5) * 5)")
            start = a.replace(minute=a.minute - a.minute % 5, second=0)
        elif b - a <= timedelta(days=3):
            bucket, step, fmt = "hour", timedelta(hours=1), "%Y-%m-%d %H:00"
            key = "strftime('%Y-%m-%d %H:00', s.time)"
            start = a.replace(minute=0, second=0)
        else:
            bucket, step, fmt = "day", timedelta(days=1), "%Y-%m-%d"
            key = "strftime('%Y-%m-%d', s.time)"
            start = a.replace(hour=0, minute=0, second=0)
        counts = {}
        for r in self._q(f"SELECT {key} AS t, COALESCE(s.vehicle_type, 'unknown') AS type, "
                         f"COUNT(*) AS n FROM sightings s LEFT JOIN watchlist w "
                         f"ON w.plate_key = s.plate_key AND s.plate_key != ''{where} "
                         f"GROUP BY t, type", args):
            counts.setdefault(r["t"], {})[r["type"]] = r["n"]
        points, t = [], start          # fill empty buckets so gaps show as zero
        while t <= b and len(points) < 500:
            label = t.strftime(fmt)
            points.append({"t": label, "counts": counts.get(label, {})})
            t += step
        return {"bucket": bucket, "points": points}

    def csv(self, filters=None):
        rows, _ = self.records(filters, limit=1_000_000)
        out = io.StringIO()
        w = csv.writer(out)
        cols = ["id", "time", "video_time", "source", "plate", "vehicle_type", "track_id",
                "confidence", "readings", "status", "reviewed", "direction", "colour", "make",
                "watch", "plate_image", "vehicle_image"]
        w.writerow(cols)
        for r in rows:
            r = dict(r)
            if r["video_time"] is not None:
                r["video_time"] = f"{r['video_time']:.2f}"
            for k in ("plate_image", "vehicle_image"):
                r[k] = f"/images/{r[k]}" if r[k] else ""
            w.writerow(["" if r.get(c) is None else r[c] for c in cols])
        return out.getvalue()

    # ---------- jobs ----------
    def add_job(self, kind, name, path):
        return self._x("INSERT INTO jobs (kind, name, path, status, created) VALUES (?, ?, ?, 'queued', ?)",
                       (kind, name, str(path), now())).lastrowid

    def update_job(self, job_id, **fields):
        self._x(f"UPDATE jobs SET {', '.join(f'{k} = ?' for k in fields)} WHERE id = ?",
                list(fields.values()) + [job_id])

    def job(self, job_id):
        rows = self._q("SELECT j.*, (SELECT COUNT(*) FROM sightings s WHERE s.job_id = j.id) AS records "
                       "FROM jobs j WHERE j.id = ?", (job_id,))
        return rows[0] if rows else None

    def jobs(self, limit=50):
        return self._q("SELECT j.*, (SELECT COUNT(*) FROM sightings s WHERE s.job_id = j.id) AS records "
                       "FROM jobs j ORDER BY j.id DESC LIMIT ?", (limit,))

    def interrupted_jobs(self):
        """Jobs left unfinished when the program last stopped."""
        return self._q("SELECT * FROM jobs WHERE status IN ('queued', 'processing') ORDER BY id")

    def delete_job(self, job_id):
        j = self.job(job_id)
        if j:
            self._x("DELETE FROM jobs WHERE id = ?", (job_id,))
        return j

    def sources(self):
        return [r["source"] for r in self._q("SELECT DISTINCT source FROM sightings ORDER BY source")]

    # ---------- cameras ----------
    def cameras(self):
        return self._q("SELECT * FROM cameras ORDER BY id")

    def add_camera(self, name, url):
        return self._x("INSERT INTO cameras (name, url, created) VALUES (?, ?, ?)",
                       (name, url, now())).lastrowid

    def camera(self, cam_id):
        rows = self._q("SELECT * FROM cameras WHERE id = ?", (cam_id,))
        return rows[0] if rows else None

    def delete_camera(self, cam_id):
        self._x("DELETE FROM cameras WHERE id = ?", (cam_id,))

    # ---------- watchlist ----------
    def watchlist(self):
        return self._q("SELECT w.*, (SELECT COUNT(*) FROM sightings s WHERE s.plate_key = w.plate_key) "
                       "AS seen, (SELECT MAX(time) FROM sightings s WHERE s.plate_key = w.plate_key) "
                       "AS last_seen FROM watchlist w ORDER BY w.kind DESC, w.plate")

    def watch(self, plate):
        """'allow', 'block' or None for a plate."""
        key = plate_key(plate)
        rows = self._q("SELECT kind FROM watchlist WHERE plate_key = ?", (key,)) if key else []
        return rows[0]["kind"] if rows else None

    def add_watch(self, plate, kind, note=""):
        key = plate_key(plate)
        self._x("INSERT INTO watchlist (plate, plate_key, kind, note, created) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(plate_key) DO UPDATE SET plate = excluded.plate, kind = excluded.kind, "
                "note = excluded.note", (plate.upper(), key, kind, note, now()))

    def delete_watch(self, watch_id):
        self._x("DELETE FROM watchlist WHERE id = ?", (watch_id,))

    def close(self):
        with self._lock:
            self._db.close()
