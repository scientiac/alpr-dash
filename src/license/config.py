"""All tunable settings in one place."""

from pathlib import Path

# ---------- Models (downloaded once, then run locally) ----------
DETECTOR_MODEL = "yolo-v9-t-512-license-plate-end2end"   # finds plates in a frame
OCR_MODEL = "cct-s-v2-global-model"                       # reads the plate text
DETECT_CONF = 0.4          # minimum detector confidence for a plate box

# ---------- Tracking and voting ----------
MATCH_IOU = 0.2            # box overlap needed to continue a track
MAX_JUMP = 1.5             # ...or centre distance, in plate widths
OCR_EVERY = 2              # min frames between OCR runs on one track
MAX_READS = 12             # keep this many best readings per track
LOST_FRAMES = 15           # frames without a detection before a track ends
MIN_HITS = 3               # detections a track needs before it is logged
MIN_READS = 2              # OCR readings needed before logging
MIN_AGREE = 0.6            # readings must agree with the voted text this much
MIN_CONF = 0.5             # OCR confidence (0-1) needed to keep a reading
MIN_LEN, MAX_LEN = 4, 10   # plausible plate text length
COOLDOWN_S = 30            # same plate seen again within this time = same pass

# ---------- Input / output ----------
TESTSET_DIR = Path(__file__).resolve().parent.parent / "testset"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
STREAM_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://")
LOG_FILE = "plates.csv"

# ---------- Dashboard ----------
HOST, PORT = "127.0.0.1", 8000
STREAM_WIDTH = 960         # preview frames are scaled to this width
JPEG_QUALITY = 75
