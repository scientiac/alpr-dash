"""Work out which videos or streams to process."""

import os
import threading
from pathlib import Path

from . import config


def is_stream(src: str) -> bool:
    return src.lower().startswith(config.STREAM_PREFIXES)


def resolve(args: list[str]) -> list[str]:
    """Return the sources to process: the given file/URL, or every test video."""
    if args:
        src = args[0]
        if not (is_stream(src) or os.path.isfile(src)):
            raise FileNotFoundError(f"{src!r} is neither a file nor a stream URL")
        return [src]

    folder = config.TESTSET_DIR
    videos = sorted(p for p in folder.iterdir() if p.suffix.lower() in config.VIDEO_EXTS) \
        if folder.is_dir() else []
    if not videos:
        raise FileNotFoundError(f"No videos found in {folder}")
    return [str(p) for p in videos]


def display_name(src: str) -> str:
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
