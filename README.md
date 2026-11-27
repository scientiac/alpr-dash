A License Plate Number Recognition Dashboard

![preview](./image.png)

[Documentation (PDF)](./docs/main.pdf)

Upload a traffic video (or connect a camera) and the system detects and tracks the
vehicles, finds and reads their number plates, and stores searchable traffic records
with photos of each plate and vehicle. Everything runs locally on the CPU.

## Install

Requirements: a 64-bit Linux, macOS or Windows computer, about 1 GB of disk space,
and an internet connection for the first run (packages and models are downloaded once).

1. Install [uv](https://docs.astral.sh/uv/):
   `curl -LsSf https://astral.sh/uv/install.sh | sh`
   (Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`)
2. In the project folder, install Python 3.12 and the packages: `uv sync`

No other system packages are needed (OpenCV includes FFmpeg; SQLite is part of Python).

## Run

```
uv run license                  # start the dashboard at http://127.0.0.1:8000
uv run license video.mp4        # ... and process a video right away
uv run license --testset        # ... and process every video in src/testset
uv run license rtsp://cam/1     # ... and add a live camera (or 0 for a webcam)
uv run license --no-web a.mp4   # no dashboard: process, print the records, exit
uv run license-eval             # measure accuracy and speed on the test set
```

The first start downloads the four models (about 280 MB). Then open the dashboard,
go to **Upload** and drop a video (MP4, AVI, MOV, MKV, WebM).

Records, photos, uploads and processed videos are kept in `data/`
(`data/plates.db` is the SQLite database). Delete that folder to start again.
All settings are in `src/license/config.py`.
