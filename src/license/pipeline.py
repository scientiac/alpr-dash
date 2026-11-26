"""The processing loop: read frames, find and read plates, track, log, draw."""

import time
from datetime import datetime, timedelta

import cv2

from . import config
from .sources import LatestFrame, display_name, is_stream
from .tracker import Tracker

GREEN, GREY = (80, 200, 60), (170, 170, 170)


def draw(frame, tracks, frame_idx):
    for t in tracks:
        if frame_idx - t.last_frame > 2:
            continue
        x1, y1, x2, y2 = t.box
        text = t.result()[0]
        color = GREEN if t.is_valid() else GREY
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"#{t.id} {text or '...'}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        ty = max(y1 - 8, th + 8)
        cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 8, ty + 4), color, -1)
        cv2.putText(frame, label, (x1 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 2, cv2.LINE_AA)


def encode(frame):
    h, w = frame.shape[:2]
    if w > config.STREAM_WIDTH:
        frame = cv2.resize(frame, (config.STREAM_WIDTH, int(h * config.STREAM_WIDTH / w)),
                           interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
    return buf.tobytes() if ok else None


def log_finished(tracks, store, name):
    for t in tracks:
        if t.is_valid():
            text, conf, n = t.result()
            store.add(t.first_seen, name, text, conf, n)


def process(src, recognizer, store, state=None, realtime=True):
    """Run the whole pipeline on one video file or stream."""
    name = display_name(src)
    stream = is_stream(src)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Could not open {name}, skipping")
        return
    print(f"Processing {name}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    reader = LatestFrame(cap) if stream else cap
    start = datetime.now()
    t0 = time.perf_counter()
    tracker = Tracker()
    frame_idx, fps, last = 0, 0.0, time.perf_counter()

    try:
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            # Wall-clock time for streams, start time + video position for files
            when = (datetime.now() if stream else
                    start + timedelta(milliseconds=cap.get(cv2.CAP_PROP_POS_MSEC)))

            boxes = recognizer.detect(frame)
            matched, finished = tracker.update(frame_idx, when, boxes)
            for t, box in matched:
                if t.needs_ocr(frame_idx):
                    t.last_ocr = frame_idx
                    r = recognizer.read(frame, box)
                    if r and r[1] >= config.MIN_CONF and config.MIN_LEN <= len(r[0]) <= config.MAX_LEN:
                        t.add_read(*r)
            log_finished(finished, store, name)

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6) if frame_idx else 0.0
            last = now

            if state is not None:
                draw(frame, tracker.tracks, frame_idx)
                active = [{"id": t.id, "plate": t.result()[0], "reads": len(t.reads)}
                          for t in tracker.tracks if frame_idx - t.last_frame <= 2]
                jpeg = encode(frame)
                if jpeg:
                    state.publish(jpeg, source=name, fps=fps, active=active)

            frame_idx += 1
            # Play files at their natural speed so the dashboard looks live
            if realtime and not stream:
                wait = t0 + frame_idx / video_fps - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
    finally:
        log_finished(tracker.flush(), store, name)
        if stream:
            reader.stop()
        cap.release()
