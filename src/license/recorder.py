"""Turn finished tracks into stored records, in a background thread.

Records come from two places:
  - a plate track that ended (its plate was read, uncertain or unreadable);
  - a moving vehicle track that ended without any plate seen on it.

For each one the recorder describes the vehicle, drops duplicates, saves the
plate and vehicle photos, and writes one row. This takes about 0.1 s per
vehicle and happens off the video loop.

Two records of one run are the same vehicle when
  - they carry the same vehicle tracking ID, or
  - the plate text is identical (within COOLDOWN_S), or
  - the vehicle looks the same (type and colour) and either the plate texts
    are mostly the same (e.g. a far-away partial reading "01KN" and the full
    "KR701KN"), or one plate sat inside the other's vehicle box at the same
    moment (e.g. a second plate or sticker on one car).
Only the best one is kept: recognized before review before unrecognized,
then the more confident reading.
"""

import queue
import threading
from dataclasses import dataclass, field

import numpy as np

from . import config
from .db import CONFUSED, now
from .vehicles import Vehicle

RANK = {"recognized": 2, "review": 1, "unrecognized": 0}


@dataclass
class Sighting:
    """Everything the video loop knows about one vehicle or plate."""
    run: int                   # which processing run it came from
    job_id: int | None
    source: str
    status: str                # recognized, review or unrecognized
    plate: str                 # '' if unreadable
    conf: float                # 0-1
    readings: int
    first_seen: float          # seconds from the start of the video
    last_seen: float
    direction: str | None
    frame: np.ndarray          # the frame used for the photos
    frame_idx: int
    plate_box: tuple | None    # None when no plate was seen
    history: dict              # frame index -> plate box
    track_id: int | None = None            # vehicle tracking ID
    vehicle_box: tuple | None = None       # vehicle box in `frame`, if known
    vehicle_type: str | None = None        # from the vehicle tracker
    vehicle_history: dict | None = None    # frame index -> vehicle box (plate-less records)


@dataclass
class Entry:
    """A stored record, remembered for a while to catch duplicates."""
    id: int
    s: Sighting
    vehicle: Vehicle
    images: tuple
    views: list = field(default_factory=list)   # (frame index, vehicle box)

    def rank(self):
        return (RANK[self.s.status], round(self.s.conf, 2), self.s.readings,
                self.s.plate_box is not None)


def _crop(frame, box, margin):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    mx, my = int(margin * (x2 - x1)), int(margin * (y2 - y1))
    return frame[max(0, y1 - my):min(h, y2 + my), max(0, x1 - mx):min(w, x2 + mx)]


def _similar(a, b):
    """Share of the shorter plate found, in order, in the other one,
    counting easily confused characters as equal."""
    a, b = a.translate(CONFUSED), b.translate(CONFUSED)
    # Longest common subsequence, by dynamic programming
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1] / max(1, min(len(a), len(b)))


def _carries(a: Entry, b: Entry):
    """Was b's plate inside a's vehicle box when a was photographed?"""
    for f, vbox in a.views:
        for d in (0, -1, 1, -2, 2, -3, 3):
            box = b.s.history.get(f + d)
            if box is not None:
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                if vbox[0] <= cx <= vbox[2] and vbox[1] <= cy <= vbox[3]:
                    return True
                break
    return False


def _rejoins(a: Entry, b: Entry):
    """Did vehicle track b start close to where and when track a ended?
    (The tracker lost the vehicle behind something and found it again.)"""
    if not (a.s.vehicle_history and b.s.vehicle_history):
        return False
    if a.s.first_seen > b.s.first_seen:
        a, b = b, a
    if not 0 <= b.s.first_seen - a.s.last_seen <= config.REJOIN_S:
        return False
    end = a.s.vehicle_history[max(a.s.vehicle_history)]
    start = b.s.vehicle_history[min(b.s.vehicle_history)]
    w = max(end[2] - end[0], start[2] - start[0], 1)
    dist = np.hypot((end[0] + end[2] - start[0] - start[2]) / 2, (end[1] + end[3] - start[1] - start[3]) / 2)
    return dist <= 0.75 * w


def same_vehicle(a: Entry, b: Entry):
    if a.s.run != b.s.run:
        return False
    if a.s.track_id is not None and a.s.track_id == b.s.track_id:
        return True
    va, vb = a.vehicle, b.vehicle
    if va.type is not None and (va.type, va.colour) == (vb.type, vb.colour) and _rejoins(a, b):
        return True
    if (a.s.first_seen > b.s.last_seen + config.COOLDOWN_S
            or b.s.first_seen > a.s.last_seen + config.COOLDOWN_S):
        return False
    if a.s.plate and a.s.plate == b.s.plate:
        return True
    va, vb = a.vehicle, b.vehicle
    if va.type is None or (va.type, va.colour) != (vb.type, vb.colour):
        return False
    if a.s.plate and b.s.plate and _similar(a.s.plate, b.s.plate) >= config.MIN_SIMILAR:
        return True
    return _carries(a, b) or _carries(b, a)


class Recorder:
    def __init__(self, db, describer, detector):
        self.db = db
        self.describer = describer
        self.detector = detector
        self._queue = queue.Queue()
        self._pending = {}                  # job id -> sightings not yet stored
        self._cond = threading.Condition()
        self._recent: list[Entry] = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def add(self, sighting: Sighting):
        with self._cond:
            self._pending[sighting.job_id] = self._pending.get(sighting.job_id, 0) + 1
        self._queue.put(sighting)

    def wait(self, job_id=None):
        """Block until the sightings of a job (or all of them) are stored."""
        with self._cond:
            self._cond.wait_for(lambda: (self._pending.get(job_id, 0) == 0) if job_id is not None
                                else not any(self._pending.values()))

    def close(self):
        """Finish the queued work, then stop."""
        if self._thread.is_alive():
            self._queue.put(None)
            self._thread.join()

    def _run(self):
        while True:
            s = self._queue.get()
            try:
                if s is None:
                    return
                self._record(s)
            except Exception as e:      # one bad record must not stop the logging
                print(f"Could not record {s.plate or 'vehicle'}: {e}")
            finally:
                if s is not None:
                    with self._cond:
                        self._pending[s.job_id] -= 1
                        self._cond.notify_all()

    def _describe(self, s: Sighting):
        box, kind = s.vehicle_box, s.vehicle_type
        if box is None and s.plate_box is not None:
            box, kind = self.detector.find(s.frame, s.plate_box)
        if box is None:
            return Vehicle()
        if self.describer is None:          # colour and make switched off
            return Vehicle(tuple(box), kind)
        return self.describer.describe(s.frame, box, kind)

    def _record(self, s: Sighting):
        v = self._describe(s)
        new = Entry(0, s, v, (None, None), [(s.frame_idx, v.box)] if v.box else [])

        # Forget entries that are too old to match anything new
        self._recent = [e for e in self._recent[-80:]
                        if e.s.run != s.run or e.s.last_seen >= s.first_seen - 3 * config.COOLDOWN_S
                        or (e.s.track_id is not None and e.s.track_id == s.track_id)]

        old = next((e for e in reversed(self._recent) if same_vehicle(e, new)), None)
        if old is not None:
            self._merge(old, new)
            return

        new.images = self._save_images(new)
        new.id = self.db.insert({**self._row(new), "time": now()})
        self._recent.append(new)
        self._print("", new)

    def _merge(self, old: Entry, new: Entry):
        """Keep the better of two records of one vehicle."""
        first = min(old.s.first_seen, new.s.first_seen)
        last = max(old.s.last_seen, new.s.last_seen)
        history = {**old.s.history, **new.s.history}
        views = old.views + new.views
        if new.rank() > old.rank():
            self.db.delete_images(*old.images)
            new.images = self._save_images(new)
            new.id = old.id
            self._recent[self._recent.index(old)] = new
            kept, dropped = new, old
        else:
            kept, dropped = old, new
        kept.s.first_seen, kept.s.last_seen, kept.s.history, kept.views = first, last, history, views
        if old.s.vehicle_history or new.s.vehicle_history:
            kept.s.vehicle_history = {**(old.s.vehicle_history or {}), **(new.s.vehicle_history or {})}
        kept.s.direction = kept.s.direction or dropped.s.direction
        kept.s.track_id = kept.s.track_id if kept.s.track_id is not None else dropped.s.track_id
        if kept.vehicle.type is None:
            kept.vehicle = dropped.vehicle
        elif kept.vehicle.type == dropped.vehicle.type:     # fill in what one photo missed
            kept.vehicle.make = kept.vehicle.make or dropped.vehicle.make
        self.db.update(kept.id, self._row(kept))
        if kept is new:
            self._print("updated ", kept)

    def _save_images(self, e: Entry):
        s = e.s
        day = now()[:10]
        name = f"{s.job_id or 0}_{s.plate or 'unread'}"
        plate = (self.db.save_image(_crop(s.frame, s.plate_box, 0.15), day, name + "_plate")
                 if s.plate_box is not None else None)
        vehicle = _crop(s.frame, e.vehicle.box, 0.05) if e.vehicle.box else s.frame
        paths = (plate, self.db.save_image(vehicle, day, name + "_vehicle"))
        s.frame = None             # photos are on disk; free the memory
        return paths

    @staticmethod
    def _row(e: Entry):
        s, v = e.s, e.vehicle
        return {
            "video_time": round(s.first_seen, 2),
            "source": s.source,
            "job_id": s.job_id,
            "track_id": s.track_id,
            "plate": s.plate,
            "confidence": round(s.conf * 100),
            "readings": s.readings,
            "status": s.status,
            "direction": s.direction,
            "vehicle_type": v.type or s.vehicle_type,
            "colour": v.colour,
            "make": v.make,
            "plate_image": e.images[0],
            "vehicle_image": e.images[1],
        }

    def _print(self, prefix, e: Entry):
        r = self._row(e)
        looks = " ".join(x for x in (r["colour"], r["make"], r["vehicle_type"]) if x) or "vehicle ?"
        m, sec = divmod(r["video_time"], 60)
        print(f"{prefix}[{int(m):02d}:{sec:04.1f}] {r['source']}: {r['plate'] or '(unreadable)'}  "
              f"{r['status']}, conf {r['confidence']}%, {looks}, track #{r['track_id'] or '-'}, "
              f"{r['direction'] or 'direction ?'}")
        if r["plate"] and self.db.watch(r["plate"]) == "block":
            print(f"  ALERT: {r['plate']} is on the blacklist")
