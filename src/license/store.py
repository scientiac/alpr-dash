"""Keep the list of recognised plates and append each one to a CSV file."""

import csv
import os
import threading

from . import config

FIELDS = ["timestamp", "source", "plate", "confidence", "readings"]


class PlateStore:
    def __init__(self, path=config.LOG_FILE):
        new = not os.path.exists(path)
        self._f = open(path, "a", newline="")
        self._w = csv.writer(self._f)
        if new:
            self._w.writerow(FIELDS)
            self._f.flush()
        self._lock = threading.Lock()
        self._last_seen = {}       # (source, plate) -> time of last log
        self.rows = []             # rows logged in this run (for the dashboard)

    def add(self, when, source, plate, conf, n):
        """Log a plate unless the same one was logged moments ago."""
        key = (source, plate)
        prev = self._last_seen.get(key)
        self._last_seen[key] = when
        if prev and abs((when - prev).total_seconds()) < config.COOLDOWN_S:
            return False
        row = {
            "timestamp": when.strftime("%Y-%m-%d %H:%M:%S"),
            "source": source,
            "plate": plate,
            "confidence": round(conf * 100),
            "readings": n,
        }
        with self._lock:
            self.rows.append(row)
            self._w.writerow([row[k] for k in FIELDS])
            self._f.flush()
        print(f"[{row['timestamp']}] {source}: {plate}  "
              f"(conf {row['confidence']}%, {n} readings)")
        return True

    def snapshot(self):
        with self._lock:
            return list(self.rows)

    def close(self):
        self._f.close()
