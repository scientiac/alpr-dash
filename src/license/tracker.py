"""Follow each plate across frames and combine its OCR readings by voting."""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

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
    first_seen: datetime
    last_frame: int
    hits: int = 1
    last_ocr: int = -10**9
    reads: list = field(default_factory=list)

    def result(self):
        return vote(self.reads)

    def needs_ocr(self, frame_idx):
        return frame_idx - self.last_ocr >= config.OCR_EVERY

    def add_read(self, text, conf):
        """Keep only the most confident readings: early ones of a far-away
        plate are often cut short, later close-up ones are sharper."""
        self.reads.append((text, conf))
        self.reads.sort(key=lambda r: r[1], reverse=True)
        del self.reads[config.MAX_READS:]

    def is_valid(self):
        text, conf, n = self.result()
        return (self.hits >= config.MIN_HITS
                and n >= config.MIN_READS
                and config.MIN_LEN <= len(text) <= config.MAX_LEN
                and agreement(self.reads, text) >= config.MIN_AGREE)


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
            t.box, t.last_frame = boxes[bi], frame_idx
            t.hits += 1
            matched.append((t, boxes[bi]))

        for bi, b in enumerate(boxes):
            if bi not in used_boxes:
                t = Track(self._next_id, b, when, frame_idx)
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
