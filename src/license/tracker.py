"""Follow each plate across frames and combine its OCR readings by voting.

Times (first_seen, last_seen) are seconds from the start of the video or stream.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from . import config


def vote(reads):
    """Weighted per-position vote.

    reads: list of (text, conf). Only readings of the most common length take
    part. Returns (text, mean_conf, n_used)."""
    if not reads:
        return "", 0.0, 0
    by_len = defaultdict(float)
    for t, c in reads:
        by_len[len(t)] += c
    length = max(by_len, key=by_len.get)
    same = [(t, c) for t, c in reads if len(t) == length]
    text = ""
    for i in range(length):
        score = defaultdict(float)
        for t, c in same:
            score[t[i]] += c
        text += max(score, key=score.get)
    return text, sum(c for _, c in same) / len(same), len(same)


def agreement(reads, text):
    """Share of characters, over all readings, that match the voted text."""
    if not reads or not text:
        return 0.0
    total = sum(
        sum(a == b for a, b in zip(t, text)) / len(text)
        for t, _ in reads if len(t) == len(text)
    )
    return total / len(reads)


def direction(history, frame_height):
    """'in', 'out' or None, from how a box moved: one that grows (or moves
    down the picture) is coming towards the camera."""
    frames = sorted(history)
    if len(frames) < 4:
        return None
    k = max(2, len(frames) // 5)              # average a few boxes at each end
    start = np.mean([history[f] for f in frames[:k]], axis=0)
    end = np.mean([history[f] for f in frames[-k:]], axis=0)
    growth = (end[2] - end[0]) / max(start[2] - start[0], 1)
    shift = ((end[1] + end[3]) - (start[1] + start[3])) / 2 / frame_height
    if growth >= config.MIN_GROWTH or (growth > 1 / config.MIN_GROWTH and shift >= config.MIN_SHIFT):
        towards = True
    elif growth <= 1 / config.MIN_GROWTH or shift <= -config.MIN_SHIFT:
        towards = False
    else:
        return None
    other = "out" if config.TOWARDS_CAMERA == "in" else "in"
    return config.TOWARDS_CAMERA if towards else other


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


@dataclass
class Track:
    id: int
    box: tuple            # x1, y1, x2, y2
    first_seen: float
    last_frame: int
    hits: int = 1
    last_ocr: int = -10**9
    last_seen: float = 0.0
    reads: list = field(default_factory=list)
    history: dict = field(default_factory=dict)   # frame index -> box
    best: tuple | None = None     # (score, frame copy, box, frame index) of the clearest reading
    largest: tuple | None = None  # (area, frame copy, box, frame index), for unreadable plates
    vehicles: Counter = field(default_factory=Counter)   # vehicle track id -> frames together

    def see(self, frame_idx, when, box):
        self.box, self.last_frame, self.last_seen = box, frame_idx, when
        self.history[frame_idx] = box

    def keep_photo(self, frame, frame_idx):
        """Remember the frame where the plate looked biggest, in case it is never read."""
        if self.best is not None:
            return
        area = (self.box[2] - self.box[0]) * (self.box[3] - self.box[1])
        if self.largest is None or area > 1.1 * self.largest[0]:
            self.largest = (area, frame.copy(), self.box, frame_idx)

    def photo(self):
        """(frame, box, frame index) for the record's photos."""
        snap = self.best or self.largest
        return snap[1:] if snap else None

    def vehicle_id(self):
        return self.vehicles.most_common(1)[0][0] if self.vehicles else None

    def result(self):
        return vote(self.reads)

    def needs_ocr(self, frame_idx):
        return frame_idx - self.last_ocr >= config.OCR_EVERY

    def add_read(self, text, conf, frame, box):
        """Keep only the most confident readings: early ones of a far-away
        plate are often cut short, later close-up ones are sharper.
        The frame of the clearest reading is kept for the photos."""
        self.reads.append((text, conf))
        self.reads.sort(key=lambda r: r[1], reverse=True)
        del self.reads[config.MAX_READS:]
        score = (round(conf, 2), box[2] - box[0])     # most certain, then largest
        if self.best is None or score > self.best[0]:
            self.best = (score, frame.copy(), box, self.last_frame)
            self.largest = None

    def direction(self, frame_height):
        return direction(self.history, frame_height)

    def is_valid(self):
        text, conf, n = self.result()
        return (self.hits >= config.MIN_HITS
                and n >= config.MIN_READS
                and config.MIN_LEN <= len(text) <= config.MAX_LEN
                and agreement(self.reads, text) >= config.MIN_AGREE)

    def status(self):
        """'recognized', 'review' (readable but a person should check it),
        'unrecognized' (seen often but never read reliably) or None (noise)."""
        if self.is_valid():
            text, conf, n = self.result()
            sure = (conf >= config.REVIEW_CONF and n >= config.REVIEW_READS
                    and agreement(self.reads, text) >= config.REVIEW_AGREE)
            return "recognized" if sure else "review"
        if self.hits >= config.UNREADABLE_MIN_HITS and self.photo() is not None:
            return "unrecognized"
        return None


class Tracker:
    def __init__(self):
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, frame_idx, when, boxes):
        """Match this frame's plate boxes to tracks.

        Returns (matched, finished): matched is a list of (track, box) for
        this frame, finished is the list of tracks that just ended."""
        pairs = []
        for bi, b in enumerate(boxes):
            bw = b[2] - b[0]
            bc = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            for t in self.tracks:
                o = iou(b, t.box)
                tc = ((t.box[0] + t.box[2]) / 2, (t.box[1] + t.box[3]) / 2)
                dist = np.hypot(bc[0] - tc[0], bc[1] - tc[1]) / max(bw, 1)
                if o >= config.MATCH_IOU or dist <= config.MAX_JUMP:
                    pairs.append((o - dist, bi, t))

        # Greedy: best-matching pairs first, each box and track used once
        pairs.sort(key=lambda p: p[0], reverse=True)
        used_boxes, used_tracks, matched = set(), set(), []
        for _, bi, t in pairs:
            if bi in used_boxes or t.id in used_tracks:
                continue
            used_boxes.add(bi)
            used_tracks.add(t.id)
            t.see(frame_idx, when, boxes[bi])
            t.hits += 1
            matched.append((t, boxes[bi]))

        for bi, b in enumerate(boxes):
            if bi not in used_boxes:
                t = Track(self._next_id, b, when, frame_idx)
                t.see(frame_idx, when, b)
                self._next_id += 1
                self.tracks.append(t)
                matched.append((t, b))

        return matched, self._expire(frame_idx)

    def _expire(self, frame_idx, force=False):
        done = [t for t in self.tracks
                if force or frame_idx - t.last_frame > config.LOST_FRAMES]
        self.tracks = [t for t in self.tracks if t not in done]
        return done

    def flush(self):
        return self._expire(0, force=True)
