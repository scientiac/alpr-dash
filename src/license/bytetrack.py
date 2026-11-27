"""Vehicle tracking with ByteTrack (Zhang et al., 2022), written with numpy only.

Each vehicle gets a track with a stable ID. A Kalman filter predicts where
every track will be in the next frame; detections are then matched to the
predictions by box overlap, in two rounds:

1. confident detections are matched to all tracks;
2. weak detections (often a vehicle that is partly hidden) are matched to
   the tracks still unmatched, which keeps them alive instead of losing them.

New tracks start only from confident detections. A track that finds no
detection is kept for TRACK_BUFFER frames before it ends, so a vehicle that
disappears behind another for a moment keeps its ID. Because the filter
predicts motion, detection does not have to run on every frame.
"""

from collections import Counter

import numpy as np

from . import config

# ---------- Kalman filter on (centre x, centre y, aspect ratio, height) ----------
_F = np.eye(8)
_F[:4, 4:] = np.eye(4)          # constant velocity: position += velocity
_H = np.eye(4, 8)               # we measure the position part only
_POS_W, _VEL_W = 1 / 20, 1 / 160


def _xyah(box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, max(y2 - y1, 1e-3)
    return np.array([x1 + w / 2, y1 + h / 2, w / h, h], float)


def _box(m):
    cx, cy, a, h = m[:4]
    w = a * h
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class _Kalman:
    def __init__(self, box):
        z = _xyah(box)
        self.x = np.r_[z, np.zeros(4)]
        h = z[3]
        std = [2 * _POS_W * h, 2 * _POS_W * h, 1e-2, 2 * _POS_W * h,
               10 * _VEL_W * h, 10 * _VEL_W * h, 1e-5, 10 * _VEL_W * h]
        self.P = np.diag(np.square(std))

    def predict(self):
        h = self.x[3]
        q = np.square([_POS_W * h, _POS_W * h, 1e-2, _POS_W * h,
                       _VEL_W * h, _VEL_W * h, 1e-5, _VEL_W * h])
        self.x = _F @ self.x
        self.P = _F @ self.P @ _F.T + np.diag(q)

    def update(self, box):
        h = self.x[3]
        r = np.diag(np.square([_POS_W * h, _POS_W * h, 1e-1, _POS_W * h]))
        s = _H @ self.P @ _H.T + r
        k = self.P @ _H.T @ np.linalg.inv(s)
        self.x = self.x + k @ (_xyah(box) - _H @ self.x)
        self.P = (np.eye(8) - k @ _H) @ self.P


def _iou_matrix(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, float)[:, None], np.asarray(b, float)[None]
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = iw * ih
    area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])
    return inter / np.maximum(area(a) + area(b) - inter, 1e-6)


def _match(tracks, dets, min_iou):
    """Greedy matching, highest overlap first. Returns (pairs, free tracks, free dets)."""
    m = _iou_matrix([t.predicted for t in tracks], [d[0] for d in dets])
    pairs, used_t, used_d = [], set(), set()
    for ti, di in sorted(np.argwhere(m >= min_iou).tolist(), key=lambda p: -m[p[0], p[1]]):
        if ti not in used_t and di not in used_d:
            used_t.add(ti)
            used_d.add(di)
            pairs.append((tracks[ti], dets[di]))
    return (pairs, [t for i, t in enumerate(tracks) if i not in used_t],
            [d for i, d in enumerate(dets) if i not in used_d])


class VehicleTrack:
    def __init__(self, tid, det, frame_idx, when):
        box, conf, label = det
        self.id = tid
        self.kf = _Kalman(box)
        self.box = self.predicted = tuple(box)
        self.labels = Counter({label: conf})
        self.hits = 1
        self.confirmed = False
        self.first_frame = self.last_frame = frame_idx
        self.first_seen = self.last_seen = when
        self.history = {frame_idx: tuple(box)}    # detected boxes only
        self.best = None                          # (area, frame copy, box, frame index)
        self.plates = set()                       # ids of plate tracks seen on it

    @property
    def label(self):
        return self.labels.most_common(1)[0][0]

    def predict(self):
        self.kf.predict()
        self.predicted = _box(self.kf.x)

    def update(self, det, frame_idx, when):
        box, conf, label = det
        self.kf.update(box)
        self.box = self.predicted = tuple(box)
        self.labels[label] += conf
        self.hits += 1
        self.confirmed = self.confirmed or self.hits >= 2
        self.last_frame, self.last_seen = frame_idx, when
        self.history[frame_idx] = tuple(box)

    def keep_photo(self, frame, frame_idx):
        """Remember the frame where the vehicle looked biggest (closest)."""
        x1, y1, x2, y2 = self.box
        area = (x2 - x1) * (y2 - y1)
        if self.best is None or area > 1.1 * self.best[0]:
            self.best = (area, frame.copy(), tuple(int(v) for v in self.box), frame_idx)

    def moved(self, frame_shape):
        """How far the vehicle travelled, as a share of the frame diagonal."""
        frames = sorted(self.history)
        (a, b) = self.history[frames[0]], self.history[frames[-1]]
        dist = np.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2)
        grow = abs((b[2] - b[0]) - (a[2] - a[0]))
        return max(dist, grow) / np.hypot(*frame_shape[:2])

    def box_near(self, frame_idx, window):
        """The detected box closest in time to a frame, if within the window."""
        for d in range(window + 1):
            for f in (frame_idx - d, frame_idx + d):
                if f in self.history:
                    return self.history[f]
        return None


class ByteTracker:
    def __init__(self):
        self.tracks: list[VehicleTrack] = []
        self._next_id = 1

    def predict(self):
        """Advance every track one frame (call on every frame)."""
        for t in self.tracks:
            t.predict()

    def update(self, dets, frame_idx, when):
        """dets: list of (box, confidence, label). Returns the tracks that ended."""
        high = [d for d in dets if d[1] >= config.TRACK_HIGH]
        low = [d for d in dets if config.TRACK_LOW <= d[1] < config.TRACK_HIGH]

        confirmed = [t for t in self.tracks if t.confirmed]
        fresh = [t for t in self.tracks if not t.confirmed]

        # 1. confident detections against confirmed tracks
        pairs, left, high = _match(confirmed, high, config.TRACK_MATCH_IOU)
        # 2. weak detections keep the remaining recent tracks alive
        recent = [t for t in left if frame_idx - t.last_frame <= 2 * config.VEHICLE_EVERY]
        pairs2, _, _ = _match(recent, low, 0.5)
        # 3. new (unconfirmed) tracks get the confident detections left over
        pairs3, unconfirmed_left, high = _match(fresh, high, config.TRACK_MATCH_IOU)
        for t, d in pairs + pairs2 + pairs3:
            t.update(d, frame_idx, when)

        # A new track that missed its second detection was probably noise
        self.tracks = [t for t in self.tracks if t not in unconfirmed_left]
        for d in high:
            if d[1] >= config.TRACK_NEW:
                self.tracks.append(VehicleTrack(self._next_id, d, frame_idx, when))
                self._next_id += 1
        return self._expire(frame_idx)

    def _expire(self, frame_idx, force=False):
        done = [t for t in self.tracks
                if force or frame_idx - t.last_frame > config.TRACK_BUFFER]
        self.tracks = [t for t in self.tracks if t not in done]
        return [t for t in done if t.confirmed]

    def flush(self):
        return self._expire(0, force=True)

    def owner(self, plate_box):
        """The confirmed vehicle track whose box holds the plate's centre (innermost)."""
        cx, cy = (plate_box[0] + plate_box[2]) / 2, (plate_box[1] + plate_box[3]) / 2
        best = None
        for t in self.tracks:
            if not t.confirmed:
                continue
            x1, y1, x2, y2 = t.predicted
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                area = (x2 - x1) * (y2 - y1)
                if best is None or area < best[0]:
                    best = (area, t)
        return best[1] if best else None
