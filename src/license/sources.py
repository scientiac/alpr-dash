"""Video inputs: files, network streams and local cameras."""

import os
import threading
from pathlib import Path

import cv2

from . import config


class VideoError(Exception):
    """A video that cannot be opened or decoded."""


def is_camera(src: str) -> bool:
    """A local camera is given by its device number, e.g. "0"."""
    return str(src).isdigit()


def is_stream(src: str) -> bool:
    """Live input (network stream or local camera): no end, no seeking."""
    return is_camera(src) or str(src).lower().startswith(config.STREAM_PREFIXES)


def open_capture(src):
    cap = cv2.VideoCapture(int(src)) if is_camera(src) else cv2.VideoCapture(str(src))
    if not cap.isOpened():
        cap.release()
        raise VideoError(f"Could not open {display_name(src)}")
    return cap


def probe(path):
    """Check that a video file can be decoded. Returns (frames, fps, duration s)."""
    cap = open_capture(path)
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise VideoError("The file has no readable frames (corrupted or not a video)")
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        return frames, fps, frames / fps if frames else 0.0
    finally:
        cap.release()


def testset():
    folder = config.TESTSET_DIR
    videos = sorted(p for p in folder.iterdir() if p.suffix.lower() in config.VIDEO_EXTS) \
        if folder.is_dir() else []
    if not videos:
        raise FileNotFoundError(f"No videos found in {folder}")
    return [str(p) for p in videos]


def resolve(args: list[str]) -> list[str]:
    """Check the files, URLs and camera numbers given on the command line."""
    for src in args:
        if not (is_stream(src) or os.path.isfile(src)):
            raise FileNotFoundError(f"{src!r} is not a file, stream URL or camera number")
    return list(args)


def display_name(src: str) -> str:
    if is_camera(src):
        return f"camera {src}"
    return src if is_stream(src) else Path(src).name


class LatestFrame:
    """Reads a live stream in a background thread and keeps only the newest
    frame, so a slow moment never makes the video fall behind."""

    def __init__(self, cap):
        self.cap = cap
        self.frame, self.ok = None, True
        self.cond = threading.Condition()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.ok:
            ok, frame = self.cap.read()
            with self.cond:
                self.ok, self.frame = ok, frame
                self.cond.notify()

    def read(self):
        with self.cond:
            self.cond.wait_for(lambda: self.frame is not None or not self.ok)
            frame, self.frame = self.frame, None
            return (frame is not None), frame

    def stop(self):
        self.ok = False
        self.thread.join(timeout=2)
