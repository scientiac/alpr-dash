"""Turn confirmed plates into stored records, in a background thread.

For each plate: find and describe its vehicle, drop duplicates, save the
plate and vehicle photos, and write one row to the database. This work is
slow-ish (about 0.2 s) but happens once per vehicle, off the video loop.

Two records are the same vehicle when they come from the same source at
about the same time and
  - the plate text is identical, or
  - the vehicle looks the same (type and colour) and either the plate texts
    are mostly the same (e.g. a far-away partial reading "01KN" and the full
    "KR701KN"), or one plate sat inside the other's vehicle box at the same
    moment (e.g. a second plate or sticker on one car).
Only the better-read one is kept.
"""

import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from . import config
from .vehicles import Vehicle

CONFUSED = str.maketrans("OIZSB", "01258")


@dataclass
class Sighting:
    """Everything the video loop knows about one confirmed plate."""
    run: int                   # which video/stream run it came from
    source: str
    plate: str
    conf: float                # 0-1
    readings: int
    first_seen: datetime
    last_seen: datetime
    direction: str | None
    frame: np.ndarray          # the frame with the clearest reading
    frame_idx: int
    plate_box: tuple
    history: dict              # frame index -> plate box


@dataclass
class Entry:
    """A stored record, remembered for a while to catch duplicates."""
    id: int
    s: Sighting
    vehicle: Vehicle
    images: tuple
    views: list = field(default_factory=list)   # (frame index, vehicle box)


def _crop(frame, box, margin):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
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


def same_vehicle(a: Entry, b: Entry):
    gap = timedelta(seconds=config.COOLDOWN_S)
    if (a.s.source != b.s.source or a.s.first_seen > b.s.last_seen + gap
            or b.s.first_seen > a.s.last_seen + gap):
        return False
    if a.s.plate == b.s.plate:
        return True
    va, vb = a.vehicle, b.vehicle
    if va.type is None or (va.type, va.colour) != (vb.type, vb.colour):
        return False
    if _similar(a.s.plate, b.s.plate) >= config.MIN_SIMILAR:
        return True
    return a.s.run == b.s.run and (_carries(a, b) or _carries(b, a))


class Recorder:
    def __init__(self, db, describer):
        self.db = db
        self.describer = describer
        self._queue = queue.Queue()
        self._recent: list[Entry] = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def add(self, sighting: Sighting):
        self._queue.put(sighting)

    def close(self):
        """Finish the queued work, then stop."""
        self._queue.put(None)
        self._thread.join()

    def _run(self):
        while (s := self._queue.get()) is not None:
            try:
                self._record(s)
            except Exception as e:      # one bad record must not stop the logging
                print(f"Could not record {s.plate}: {e}")

    def _record(self, s: Sighting):
        v = self.describer.describe(s.frame, s.plate_box)
        new = Entry(0, s, v, (None, None), [(s.frame_idx, v.box)] if v.box else [])

        # Forget entries that are too old to match anything new
        horizon = s.first_seen - timedelta(seconds=3 * config.COOLDOWN_S)
        self._recent = [e for e in self._recent[-50:] if e.s.last_seen >= horizon]

        old = next((e for e in reversed(self._recent) if same_vehicle(e, new)), None)
        if old is not None:
            self._merge(old, new)
            return

        new.images = self._save_images(new)
        new.id = self.db.insert(self._row(new))
        self._recent.append(new)
        self._print("", new)

    def _merge(self, old: Entry, new: Entry):
        """Keep the better-read of two records of one vehicle."""
        first = min(old.s.first_seen, new.s.first_seen)
        last = max(old.s.last_seen, new.s.last_seen)
        history = {**old.s.history, **new.s.history}
        views = old.views + new.views
        if (round(new.s.conf, 2), new.s.readings) > (round(old.s.conf, 2), old.s.readings):
            self.db.delete_images(*old.images)
            new.images = self._save_images(new)
            new.id = old.id
            self._recent[self._recent.index(old)] = new
            kept, dropped = new, old
        else:
            kept, dropped = old, new
        kept.s.first_seen, kept.s.last_seen, kept.s.history, kept.views = first, last, history, views
        kept.s.direction = kept.s.direction or dropped.s.direction
        if kept.vehicle.type == dropped.vehicle.type:     # fill in what one photo missed
            kept.vehicle.make = kept.vehicle.make or dropped.vehicle.make
        self.db.update(kept.id, self._row(kept))
        print(f"  (same vehicle as {kept.s.plate}: dropped {dropped.s.plate})")
        if kept is new:
            self._print("updated ", kept)

    def _save_images(self, e: Entry):
        s = e.s
        day = s.first_seen.strftime("%Y-%m-%d")
        name = f"{s.first_seen:%H%M%S}_{s.plate}"
        plate = _crop(s.frame, s.plate_box, 0.15)
        vehicle = _crop(s.frame, e.vehicle.box, 0.05) if e.vehicle.box else s.frame
        paths = (self.db.save_image(plate, day, name + "_plate"),
                 self.db.save_image(vehicle, day, name + "_vehicle"))
        s.frame = None             # photos are on disk; free the memory
        return paths

    @staticmethod
    def _row(e: Entry):
        s, v = e.s, e.vehicle
        return {
            "time": s.first_seen.strftime("%Y-%m-%d %H:%M:%S"),
            "source": s.source,
            "plate": s.plate,
            "confidence": round(s.conf * 100),
            "readings": s.readings,
            "direction": s.direction,
            "vehicle_type": v.type,
            "colour": v.colour,
            "make": v.make,
            "plate_image": e.images[0],
            "vehicle_image": e.images[1],
        }

    @staticmethod
    def _print(prefix, e: Entry):
        r = Recorder._row(e)
        looks = " ".join(x for x in (r["colour"], r["make"], r["vehicle_type"]) if x) or "vehicle ?"
        print(f"{prefix}[{r['time']}] {r['source']}: {r['plate']}  "
              f"(conf {r['confidence']}%, {r['readings']} readings)  "
              f"{looks}, {r['direction'] or 'direction ?'}")
