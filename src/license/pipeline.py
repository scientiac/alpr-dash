"""The processing loop for one video, stream or camera.

Per frame:
  1. read the frame;
  2. detect vehicles (every VEHICLE_EVERY frames) and track them (ByteTrack);
  3. detect plates and follow them with the plate tracker;
  4. link each plate to the vehicle it sits on;
  5. read plates (OCR) and vote over the readings;
  6. pass tracks that ended to the recorder (it runs in the background);
  7. draw the boxes, write the annotated video and update the live view.
"""

import itertools
import queue
import threading
import time
from dataclasses import dataclass

import cv2

from . import config
from .bytetrack import ByteTracker
from .recorder import Sighting
from .sources import LatestFrame, VideoError, display_name, is_stream, open_capture
from .tracker import Tracker, direction

GREEN, GREY, BLUE = (80, 200, 60), (170, 170, 170), (237, 140, 51)
_runs = itertools.count(1)
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)   # hide FFmpeg codec notices


@dataclass
class Models:
    recognizer: object      # plate detector + OCR
    vehicles: object        # vehicle detector
    describer: object       # colour, make, van


class Feed:
    """The live view of one running job, shared with the web server."""

    def __init__(self):
        self.cond = threading.Condition()
        self.jpeg = None
        self.frame_no = 0
        self.info = {"source": "", "fps": 0.0, "active": [], "vehicles": 0, "done": False}

    def publish(self, jpeg, **info):
        with self.cond:
            self.jpeg = jpeg
            self.frame_no += 1
            self.info.update(info)
            self.cond.notify_all()

    def finish(self):
        with self.cond:
            self.info["done"] = True
            self.cond.notify_all()


class AnnotatedWriter:
    """Writes the annotated video (WebM/VP8, plays in browsers) in its own
    thread, so encoding overlaps with detection."""

    def __init__(self, path, fps):
        self.path = path
        self._fps = fps
        self._writer = None
        self._queue = queue.Queue(maxsize=16)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def write(self, frame):
        self._queue.put(frame)

    def _run(self):
        while (frame := self._queue.get()) is not None:
            frame = _scaled(frame)
            if self._writer is None:
                h, w = frame.shape[:2]
                self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"VP80"),
                                               self._fps, (w, h))
            self._writer.write(frame)

    def close(self):
        self._queue.put(None)
        self._thread.join()
        if self._writer is not None:
            self._writer.release()
        return self._writer is not None


def _scaled(frame):
    h, w = frame.shape[:2]
    if w > config.STREAM_WIDTH:
        frame = cv2.resize(frame, (config.STREAM_WIDTH, int(h * config.STREAM_WIDTH / w) // 2 * 2),
                           interpolation=cv2.INTER_AREA)
    return frame


def _label(frame, x, y, text, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y = max(y - 6, th + 8)
    cv2.rectangle(frame, (x, y - th - 6), (x + tw + 8, y + 4), color, -1)
    cv2.putText(frame, text, (x + 4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


def draw(frame, plates, vehicles, frame_idx):
    for v in vehicles.tracks:
        if v.confirmed and frame_idx - v.last_frame <= 2 * config.VEHICLE_EVERY:
            x1, y1, x2, y2 = (int(c) for c in v.predicted)
            cv2.rectangle(frame, (x1, y1), (x2, y2), BLUE, 2)
            _label(frame, x1, y1, f"#{v.id} {v.label}", BLUE)
    for t in plates.tracks:
        if frame_idx - t.last_frame > 3:
            continue
        x1, y1, x2, y2 = t.box
        text = t.result()[0]
        color = GREEN if t.is_valid() else GREY
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        _label(frame, x1, y2 + 26, text or "...", color)


def encode(frame):
    ok, buf = cv2.imencode(".jpg", _scaled(frame), [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
    return buf.tobytes() if ok else None


class Run:
    """State of one processing run; turns ended tracks into sightings."""

    def __init__(self, recorder, name, job_id):
        self.recorder, self.name, self.job_id = recorder, name, job_id
        self.id = next(_runs)
        self.all_vehicles = {}            # vehicle track id -> track (also ended ones)

    def plates_ended(self, tracks):
        for t in tracks:
            status = t.status()
            if status is None:
                continue
            frame, box, idx = t.photo()
            text, conf, n = t.result() if status != "unrecognized" else ("", 0.0, 0)
            vt = self.all_vehicles.get(t.vehicle_id())
            self.recorder.add(Sighting(
                self.id, self.job_id, self.name, status, text, conf, n, t.first_seen, t.last_seen,
                t.direction(frame.shape[0]), frame, idx, box, t.history,
                track_id=vt.id if vt else None,
                vehicle_box=vt.box_near(idx, config.VEHICLE_EVERY) if vt else None,
                vehicle_type=vt.label if vt else None))

    def vehicles_ended(self, tracks, frame_shape):
        """Moving vehicles become records too, so vehicles whose plate was never
        seen are counted (as unrecognized). If a plate was read on the vehicle,
        the recorder merges the two by tracking ID and keeps the plate."""
        for v in tracks:
            snap, v.best = v.best, None           # free the frame copy
            if (not config.RECORD_VEHICLES_WITHOUT_PLATE or snap is None
                    or v.hits < config.MIN_VEHICLE_HITS
                    or v.moved(frame_shape) < config.MIN_VEHICLE_MOVE
                    or snap[0] < config.MIN_VEHICLE_AREA * frame_shape[0] * frame_shape[1]):
                continue
            _, frame, box, idx = snap
            self.recorder.add(Sighting(
                self.id, self.job_id, self.name, "unrecognized", "", 0.0, 0, v.first_seen,
                v.last_seen, direction(v.history, frame_shape[0]), frame, idx, None, {},
                track_id=v.id, vehicle_box=box, vehicle_type=v.label, vehicle_history=v.history))


def process(src, models, recorder, *, name=None, job_id=None, feed=None, realtime=False,
            stop=None, progress=None, output=None):
    """Run the whole pipeline on one video file, stream or camera.

    name: shown in the records (default: the file name);
    feed: Feed for the live view; stop: threading.Event to cancel;
    progress(fraction, frames, fps): called about twice a second;
    output: path of the annotated video to write (None = do not write).
    Returns a summary dict. Raises VideoError if the input cannot be read."""
    name = name or display_name(src)
    stream = is_stream(src)
    cap = open_capture(src)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if not stream else 0
    reader = LatestFrame(cap) if stream else cap
    writer = AnnotatedWriter(output, video_fps) if output else None
    print(f"Processing {name}")

    plates, vehicles = Tracker(), ByteTracker()
    every = config.VEHICLE_EVERY_LIVE if stream else config.VEHICLE_EVERY
    run = Run(recorder, name, job_id)
    t0 = time.perf_counter()
    frame_idx, frame_time, last, behind, shape = 0, 0.0, t0, False, None
    last_report = 0.0

    try:
        while not (stop and stop.is_set()):
            ok, frame = reader.read()
            if not ok:
                break
            shape = frame.shape
            if stream:
                when = time.perf_counter() - t0
            else:
                when = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 or frame_idx / video_fps

            # When real-time playback falls behind, skip detection on this
            # frame; the tracks keep their last position
            vehicles.predict()
            if not behind:
                if frame_idx % every == 0:
                    ended = vehicles.update(models.vehicles.detect(frame), frame_idx, when)
                    for v in vehicles.tracks:
                        run.all_vehicles[v.id] = v
                        if v.confirmed and v.last_frame == frame_idx:
                            v.keep_photo(frame, frame_idx)
                    run.vehicles_ended(ended, shape)

                matched, finished = plates.update(frame_idx, when, models.recognizer.detect(frame))
                for t, box in matched:
                    owner = vehicles.owner(box)
                    if owner is not None:
                        t.vehicles[owner.id] += 1
                        owner.plates.add(t.id)
                    if t.needs_ocr(frame_idx):
                        t.last_ocr = frame_idx
                        r = models.recognizer.read(frame, box)
                        if (r and r[1] >= config.MIN_CONF
                                and config.MIN_LEN <= len(r[0]) <= config.MAX_LEN):
                            t.add_read(*r, frame, box)
                    t.keep_photo(frame, frame_idx)
                run.plates_ended(finished)

            now = time.perf_counter()
            frame_time = 0.95 * frame_time + 0.05 * (now - last) if frame_idx else 1 / video_fps
            fps, last = 1 / max(frame_time, 1e-6), now
            frame_idx += 1

            if writer is not None or feed is not None:
                draw(frame, plates, vehicles, frame_idx - 1)
            if writer is not None:
                writer.write(frame)
            if feed is not None:
                jpeg = encode(frame)
                if jpeg:
                    active = [{"id": t.id, "plate": t.result()[0], "reads": len(t.reads)}
                              for t in plates.tracks if frame_idx - 1 - t.last_frame <= 3]
                    live = sum(1 for v in vehicles.tracks
                               if v.confirmed and frame_idx - 1 - v.last_frame <= 2 * config.VEHICLE_EVERY)
                    feed.publish(jpeg, source=name, fps=fps, active=active, vehicles=live,
                                 progress=frame_idx / total if total else None)
            if progress and now - last_report > 0.5:
                last_report = now
                progress(min(frame_idx / total, 1.0) if total else 0.0, frame_idx,
                         frame_idx / (now - t0))

            # Play files at their natural speed when asked (looks live)
            if realtime and not stream:
                wait = t0 + frame_idx / video_fps - time.perf_counter()
                behind = wait < 0 and not behind    # never skip twice in a row
                if wait > 0:
                    time.sleep(wait)
    finally:
        run.plates_ended(plates.flush())
        if shape is not None:
            run.vehicles_ended(vehicles.flush(), shape)
        if stream:
            reader.stop()
        cap.release()
        wrote = writer.close() if writer else False
        if feed is not None:
            feed.finish()

    if frame_idx == 0:
        raise VideoError("No frames could be decoded (the file may be corrupted)")
    elapsed = time.perf_counter() - t0
    return {"frames": frame_idx, "total": total, "fps": frame_idx / elapsed,
            "seconds": elapsed, "duration": frame_idx / video_fps, "annotated": wrote,
            "cut_short": bool(total) and frame_idx < 0.95 * total
                         and not (stop and stop.is_set())}
