import csv
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import imutils
import numpy as np
import pytesseract

# ---------- Settings ----------
# Plate size as a fraction of frame width (measured on 1280 px footage: ~0.074)
PLATE_W_MIN, PLATE_W_MAX = 0.04, 0.13
MIN_RATIO, MAX_RATIO = 2.0, 6.0
MAX_TILT = 15
ROI_TOP, ROI_BOTTOM = 0.20, 0.80   # ignore sky/background and bottom pavement
PROCESS_WIDTH = None               # e.g. 1280 to downscale big frames

# Filled in by configure() from the real frame width
MIN_AREA = MAX_AREA = 0
KERNEL_W = 13

WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
TARGET_H = 160
TEXT_H = 60

MIN_CHARS = 6              # characters needed to accept a candidate as a plate
MIN_LEN, MAX_LEN = 4, 10   # plausible plate text length
MIN_HITS = 4               # detections a track needs before it is drawn or logged
MIN_READS = 3              # OCR readings needed before logging
MIN_AGREE = 0.6            # readings must agree with the voted text at least this much
MIN_CONF = 40              # Tesseract confidence needed to keep a reading
MAX_GREEN = 0.35           # reject candidates that are mostly green (foliage)

DETECT_EVERY = 2     # run plate detection every N frames
OCR_EVERY = 4        # min frames between OCR runs on the same track
MAX_READS = 8        # stop OCR-ing a track after this many readings
LOST_FRAMES = 20     # frames without a detection before a track is finished
COOLDOWN_S = 30      # don't re-log the same plate within this many seconds
LOG_FILE = "plates.csv"
DEBUG = False

TESTSET_DIR = Path(__file__).resolve().parent.parent / "testset"   # src/testset
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
STREAM_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://")


# ---------- Detection ----------
def configure(frame_w):
    """Derive pixel limits from the frame width."""
    global MIN_AREA, MAX_AREA, KERNEL_W
    w_min, w_max = PLATE_W_MIN * frame_w, PLATE_W_MAX * frame_w
    MIN_AREA = w_min ** 2 / 3.5
    MAX_AREA = w_max ** 2 / 2.0
    KERNEL_W = max(5, int(frame_w * 0.010)) | 1
    print(f"Frame width {frame_w}: plate area {MIN_AREA:.0f}-{MAX_AREA:.0f} px², "
          f"kernel {KERNEL_W}")


def normalize(rect):
    (cx, cy), (w, h), angle = rect
    if w < h:
        w, h = h, w
        angle += 90
    angle = ((angle + 90) % 180) - 90
    return (cx, cy), (w, h), angle


def validate(rect):
    _, (w, h), angle = normalize(rect)
    if w == 0 or h == 0:
        return False
    return (
        MIN_AREA <= w * h <= MAX_AREA
        and MIN_RATIO <= w / h <= MAX_RATIO
        and abs(angle) <= MAX_TILT
    )


def detect_candidates(img):
    gray = cv2.cvtColor(cv2.GaussianBlur(img, (3, 3), 0), cv2.COLOR_BGR2GRAY)
    sobel_x = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    sobel_x = np.uint8(255 * sobel_x / max(sobel_x.max(), 1))
    _, binary = cv2.threshold(sobel_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (KERNEL_W, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cnts = imutils.grab_contours(
        cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    )
    return [r for r in map(cv2.minAreaRect, cnts) if validate(r)]


# ---------- Cropping ----------
def crop_rotated(img, rect, pad_x=0.08, pad_y=0.25):
    (cx, cy), (w, h), angle = normalize(rect)
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    size = (int(w * (1 + 2 * pad_x)), int(h * (1 + 2 * pad_y)))
    return cv2.getRectSubPix(rotated, size, (cx, cy))


def group_chars(binary):
    H, _ = binary.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (0.25 * H <= h <= 0.95 * H):
            continue
        if not (0.08 * h <= w <= 1.1 * h):
            continue
        if area < 0.15 * w * h:
            continue
        comps.append((x, y, w, h))
    if len(comps) < MIN_CHARS:
        return None

    med = np.median([c[3] for c in comps])
    comps = [c for c in comps if 0.85 * med <= c[3] <= 1.15 * med]
    if len(comps) < MIN_CHARS:
        return None

    # Group into rows by vertical centre
    comps.sort(key=lambda c: c[1] + c[3] / 2)
    rows = [[comps[0]]]
    for c in comps[1:]:
        cy = c[1] + c[3] / 2
        row_cy = np.mean([r[1] + r[3] / 2 for r in rows[-1]])
        if cy - row_cy > 0.5 * med:
            rows.append([c])
        else:
            rows[-1].append(c)

    # Each row must sit on a common line with even spacing
    good = []
    for r in rows:
        if len(r) < 2:
            continue
        r.sort(key=lambda c: c[0])
        tops = np.array([c[1] for c in r], dtype=float)
        bots = np.array([c[1] + c[3] for c in r], dtype=float)
        if tops.std() > 0.12 * med or bots.std() > 0.12 * med:
            continue
        gaps = [r[i + 1][0] - (r[i][0] + r[i][2]) for i in range(len(r) - 1)]
        if max(gaps) > 1.5 * med or min(gaps) < -0.3 * med:
            continue
        good.append(r)

    rows = good
    kept = [c for r in rows for c in r]
    if len(kept) < MIN_CHARS:
        return None

    hs = np.array([c[3] for c in kept], dtype=float)
    uniform = 1 - min(hs.std() / hs.mean(), 0.5)
    return {
        "box": (
            min(c[0] for c in kept),
            min(c[1] for c in kept),
            max(c[0] + c[2] for c in kept),
            max(c[1] + c[3] for c in kept),
        ),
        "n": len(kept),
        "rows": len(rows),
        "char_h": float(np.median(hs)),
        "score": len(kept) * uniform,
    }


def locate_text(crop):
    if crop.shape[0] < 8 or crop.shape[1] < 8:
        return None

    # Foliage rejection
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    green = ((hsv[..., 0] > 35) & (hsv[..., 0] < 85) & (hsv[..., 1] > 60)).mean()
    if green > MAX_GREEN:
        return None

    scale = TARGET_H / crop.shape[0]
    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    soft = cv2.GaussianBlur(gray, (3, 3), 0)
    H, W = gray.shape
    block = (H // 2) | 1

    best = None
    for pol in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
        _, otsu = cv2.threshold(soft, 0, 255, pol + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(
            soft, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, pol, block, 10
        )
        for binary in (otsu, adaptive):
            res = group_chars(binary)
            if res and (best is None or res["score"] > best["score"]):
                best = res
    if best is None:
        return None

    x0, y0, x1, y1 = best["box"]
    m = int(0.15 * (y1 - y0))
    best["gray"] = gray[max(0, y0 - m):min(H, y1 + m), max(0, x0 - m):min(W, x1 + m)]
    return best


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def find_plates(img):
    """All plates in the frame (best-scoring first), overlaps removed."""
    y0 = int(img.shape[0] * ROI_TOP)
    y1 = int(img.shape[0] * ROI_BOTTOM)
    roi = img[y0:y1]

    found_all = []
    for rect in detect_candidates(roi):
        found = locate_text(crop_rotated(roi, rect))
        if found:
            x, y, w, h = cv2.boundingRect(cv2.boxPoints(rect).astype(np.int32))
            found["bbox"] = (x, y + y0, w, h)   # back to full-frame coordinates
            found_all.append(found)

    found_all.sort(key=lambda f: f["score"], reverse=True)
    kept = []
    for f in found_all:
        if all(iou(f["bbox"], k["bbox"]) < 0.3 for k in kept):
            kept.append(f)
    return kept


# ---------- OCR ----------
def run_ocr(img, psm):
    cfg = f"--oem 1 --psm {psm} -c tessedit_char_whitelist={WHITELIST}"
    data = pytesseract.image_to_data(
        img, config=cfg, output_type=pytesseract.Output.DICT
    )
    words, confs = [], []
    for t, c in zip(data["text"], data["conf"]):
        t = re.sub(r"[^A-Za-z0-9]", "", t).upper()
        if t and float(c) >= 0:
            words.append(t)
            confs.append(float(c))
    if not words:
        return "", 0.0
    return "".join(words), sum(confs) / len(confs)


def vote(results):
    """Weighted per-position vote among results of the best-supported length.
    results: list of (text, conf). Returns (text, mean_conf, n_used)."""
    if not results:
        return "", 0.0, 0
    weight = defaultdict(float)
    for t, c in results:
        weight[len(t)] += c
    length = max(weight, key=weight.get)
    same = [(t, c) for t, c in results if len(t) == length]
    best = ""
    for i in range(length):
        score = defaultdict(float)
        for t, c in same:
            score[t[i]] += c
        best += max(score, key=score.get)
    return best, sum(c for _, c in same) / len(same), len(same)


def read_plate(found):
    """One OCR reading of one plate crop -> (text, confidence)."""
    gray = found["gray"]
    scale = TEXT_H / found["char_h"]
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interp)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(otsu) < 127:
        otsu = cv2.bitwise_not(otsu)
    adaptive = cv2.adaptiveThreshold(
        clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        (int(TEXT_H * 0.75)) | 1, 10,
    )
    if np.mean(adaptive) < 127:
        adaptive = cv2.bitwise_not(adaptive)

    psm = 7 if found["rows"] == 1 else 6
    results = []
    for name, v in (("clahe", clahe), ("otsu", otsu), ("adaptive", adaptive)):
        v = cv2.copyMakeBorder(v, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
        text, conf = run_ocr(v, psm)
        if DEBUG:
            print(f"  {name:9s} psm{psm} -> {text!r} ({conf:.0f})")
        if len(text) >= MIN_LEN:
            results.append((text, conf))

    text, conf, _ = vote(results)
    return text, conf


# ---------- Tracking ----------
class Track:
    _next_id = 1

    def __init__(self, bbox, frame_idx, when):
        self.id = Track._next_id
        Track._next_id += 1
        self.bbox = bbox
        self.last_seen = frame_idx
        self.last_ocr = -10**9
        self.first_seen = when
        self.reads = []
        self.hits = 0

    def consensus(self, min_reads=1):
        text, conf, n = vote(self.reads)
        return (text, conf, n) if n >= min_reads else ("", 0.0, n)

    def agreement(self, text):
        """Mean per-position match between all readings and the voted text."""
        if not self.reads or not text:
            return 0.0
        total = 0.0
        for t, _ in self.reads:
            if len(t) == len(text):
                total += sum(a == b for a, b in zip(t, text)) / len(text)
        return total / len(self.reads)


class Tracker:
    def __init__(self):
        self.tracks = []

    def update(self, frame_idx, when, detections):
        if detections:
            taken = set()
            for det in detections:
                x, y, w, h = det["bbox"]
                cx, cy = x + w / 2, y + h / 2
                best, best_d = None, 1.5 * w          # max jump between detections
                for t in self.tracks:
                    if t.id in taken:
                        continue
                    tx, ty, tw, th = t.bbox
                    d = np.hypot(cx - (tx + tw / 2), cy - (ty + th / 2))
                    if d < best_d:
                        best, best_d = t, d
                if best is None:
                    best = Track(det["bbox"], frame_idx, when)
                    self.tracks.append(best)
                taken.add(best.id)
                best.bbox = det["bbox"]
                best.last_seen = frame_idx
                best.hits += 1

                # Only OCR tracks that have persisted for at least 2 detections
                if (best.hits >= 2 and len(best.reads) < MAX_READS
                        and frame_idx - best.last_ocr >= OCR_EVERY):
                    text, conf = read_plate(det)
                    best.last_ocr = frame_idx
                    if text and conf >= MIN_CONF:
                        best.reads.append((text, conf))

        return self._expire(frame_idx, force=False)

    def _expire(self, frame_idx, force):
        done, alive = [], []
        for t in self.tracks:
            if force or frame_idx - t.last_seen > LOST_FRAMES:
                done.append(t)
            else:
                alive.append(t)
        self.tracks = alive

        events = []
        for t in done:
            if t.hits < MIN_HITS:
                continue
            text, conf, n = t.consensus(MIN_READS)
            if (text and MIN_LEN <= len(text) <= MAX_LEN
                    and conf >= MIN_CONF and t.agreement(text) >= MIN_AGREE):
                events.append((t.first_seen, text, conf, n))
        return events

    def flush(self, frame_idx):
        return self._expire(frame_idx, force=True)


# ---------- Logging ----------
class PlateLog:
    def __init__(self, path):
        new = not os.path.exists(path)
        self.f = open(path, "a", newline="")
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(["timestamp", "source", "plate", "confidence", "readings"])
        self.last = {}

    def add(self, when, text, conf, n, source):
        key = (source, text)
        prev = self.last.get(key)
        if prev and (when - prev).total_seconds() < COOLDOWN_S:
            return
        self.last[key] = when
        stamp = when.strftime("%Y-%m-%d %H:%M:%S")
        self.w.writerow([stamp, source, text, f"{conf:.0f}", n])
        self.f.flush()
        print(f"[{stamp}] {source}: {text}  (conf {conf:.0f}, {n} readings)")

    def close(self):
        self.f.close()


# ---------- Entry point ----------
def list_test_videos():
    if not TESTSET_DIR.is_dir():
        return []
    return sorted(p for p in TESTSET_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTS)


def process_source(src, is_stream, log, show):
    """Run the pipeline on one video or stream. Returns True if the user quit."""
    name = src if is_stream else Path(src).name
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Could not open {name}, skipping")
        return False
    print(f"Processing {name}")

    start = datetime.now()
    tracker = Tracker()
    frame_idx, last_logged, quit_ = 0, "", False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if PROCESS_WIDTH and frame.shape[1] > PROCESS_WIDTH:
                frame = imutils.resize(frame, width=PROCESS_WIDTH)
            if frame_idx == 0:
                configure(frame.shape[1])

            # Wall-clock time for streams, start time + video position for files
            when = (datetime.now() if is_stream else
                    start + timedelta(milliseconds=cap.get(cv2.CAP_PROP_POS_MSEC)))

            detections = find_plates(frame) if frame_idx % DETECT_EVERY == 0 else None
            for ev in tracker.update(frame_idx, when, detections):
                log.add(*ev, source=name)
                last_logged = f"{ev[0]:%H:%M:%S}  {ev[1]}"

            if show:
                for t in tracker.tracks:
                    if t.hits < MIN_HITS or frame_idx - t.last_seen > DETECT_EVERY * 2:
                        continue
                    x, y, w, h = t.bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = t.consensus()[0] or "..."
                    cv2.putText(frame, f"#{t.id} {label}", (x, max(y - 8, 18)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if last_logged:
                    cv2.putText(frame, f"Last: {last_logged}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                cv2.imshow("License plates (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    quit_ = True
                    break
            frame_idx += 1
    finally:
        for ev in tracker.flush(frame_idx):
            log.add(*ev, source=name)
        cap.release()
    return quit_


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show = "--no-display" not in sys.argv

    if args:
        src = args[0]
        if src.lower().startswith(STREAM_PREFIXES):
            jobs = [(src, True)]
        elif os.path.isfile(src):
            jobs = [(src, False)]
        else:
            raise FileNotFoundError(f"{src!r} is neither a file nor a stream URL")
    else:
        videos = list_test_videos()
        if not videos:
            raise FileNotFoundError(f"No videos found in {TESTSET_DIR}")
        jobs = [(str(p), False) for p in videos]
        print(f"Using {len(jobs)} video(s) from {TESTSET_DIR}")

    log = PlateLog(LOG_FILE)
    try:
        for src, is_stream in jobs:
            if process_source(src, is_stream, log, show):
                break
    finally:
        log.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
