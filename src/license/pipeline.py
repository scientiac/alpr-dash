"""The processing loop: read frames, find and read plates, track, log, draw."""

import itertools
import time
from datetime import datetime, timedelta

import cv2

from . import config
from .recorder import Sighting
from .sources import LatestFrame, display_name, is_stream
from .tracker import Tracker

GREEN, GREY = (80, 200, 60), (170, 170, 170)
_runs = itertools.count(1)


def draw(frame, tracks, frame_idx):
    for t in tracks:
        if frame_idx - t.last_frame > 3:
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


def log_finished(tracks, recorder, name, run):
    """Hand every confirmed plate to the recorder (which works in the background)."""
    for t in tracks:
        if t.is_valid() and t.best is not None:
            text, conf, n = t.result()
            _, frame, box, idx = t.best
            recorder.add(Sighting(run, name, text, conf, n, t.first_seen, t.last_seen,
                                  t.direction(frame.shape[0]), frame, idx, box, t.history))


def process(src, recognizer, recorder, state=None, realtime=True):
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
    run = next(_runs)
    frame_idx, frame_time, last, behind = 0, 0.0, time.perf_counter(), False

    try:
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            # Wall-clock time for streams, start time + video position for files
            when = (datetime.now() if stream else
                    start + timedelta(milliseconds=cap.get(cv2.CAP_PROP_POS_MSEC)))

            # When playback falls behind, skip detection on this frame;
            # the tracks simply keep their last box
            if not behind:
                boxes = recognizer.detect(frame)
                matched, finished = tracker.update(frame_idx, when, boxes)
                for t, box in matched:
                    if t.needs_ocr(frame_idx):
                        t.last_ocr = frame_idx
                        r = recognizer.read(frame, box)
                        if (r and r[1] >= config.MIN_CONF
                                and config.MIN_LEN <= len(r[0]) <= config.MAX_LEN):
                            t.add_read(*r, frame, box)
                log_finished(finished, recorder, name, run)

            now = time.perf_counter()
            frame_time = 0.95 * frame_time + 0.05 * (now - last) if frame_idx else 1 / video_fps
            fps, last = 1 / max(frame_time, 1e-6), now

            if state is not None:
                draw(frame, tracker.tracks, frame_idx)
                active = [{"id": t.id, "plate": t.result()[0], "reads": len(t.reads)}
                          for t in tracker.tracks if frame_idx - t.last_frame <= 3]
                jpeg = encode(frame)
                if jpeg:
                    state.publish(jpeg, source=name, fps=fps, active=active)

            frame_idx += 1
            # Play files at their natural speed so the dashboard looks live
            if realtime and not stream:
                wait = t0 + frame_idx / video_fps - time.perf_counter()
                behind = wait < 0 and not behind    # never skip twice in a row
                if wait > 0:
                    time.sleep(wait)
    finally:
        log_finished(tracker.flush(), recorder, name, run)
        if stream:
            reader.stop()
        cap.release()
