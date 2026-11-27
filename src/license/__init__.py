"""Automatic number plate recognition with a web dashboard.

Usage:
  license                          start the dashboard; upload videos there
  license <file|url|camera> ...    also process these (a URL or camera number,
                                   such as 0, is added as a live camera)
  license --testset                also process every video in src/testset once
  license --test                   play all test videos in a loop, like a live
                                   camera (a demonstration mode)
Options:
  --no-web       no dashboard: process the given files, print the records, exit
  --realtime     play files at their natural speed (looks like a live camera)
  --port N       dashboard port (default 8000)
  --host H       dashboard address (default 127.0.0.1, this computer only)
"""

import sys
import time

from . import config, sources
from .db import Database


def _option(argv, name, default):
    if name in argv:
        i = argv.index(name)
        if i + 1 >= len(argv):
            sys.exit(f"{name} needs a value")
        value = argv[i + 1]
        del argv[i:i + 2]
        return value
    return default


def load_models():
    """Load (and on the first run download) all models."""
    from .pipeline import Models
    from .recognizer import Recognizer
    from .vehicles import Describer, VehicleDetector
    return Models(Recognizer(), VehicleDetector(),
                  Describer() if config.DESCRIBE_VEHICLES else None)


def main() -> None:
    argv = sys.argv[1:]
    if "-h" in argv or "--help" in argv:
        print(__doc__)
        return
    port = int(_option(argv, "--port", config.PORT))
    host = _option(argv, "--host", config.HOST)
    web = "--no-web" not in argv
    args = [a for a in argv if not a.startswith("--")]
    try:
        inputs = sources.resolve(args) + (sources.testset() if "--testset" in argv else [])
    except FileNotFoundError as e:
        sys.exit(str(e))
    if not web and not inputs and "--test" not in argv:
        sys.exit("Nothing to do: give a video file (or --testset / --test) with --no-web")

    db = Database()
    print(f"Recording to {config.DB_FILE} (photos in {config.IMAGE_DIR})")
    app = None
    if web:
        from .web.server import App, serve
        app = App(db)
        serve(app, host, port)

    print("Loading models...")
    try:
        models = load_models()
    except Exception as e:      # no internet on the first run, broken cache, ...
        msg = (f"Could not load the models: {e}. The first run downloads them, "
               "so it needs an internet connection.")
        print(msg)
        if app is None:
            sys.exit(1)
        app.error = msg
        models = None

    from .jobs import JobManager
    from .recorder import Recorder
    recorder = jobs = None
    if models is not None:
        recorder = Recorder(db, models.describer, models.vehicles)
        jobs = JobManager(db, models, recorder)
        jobs.realtime = "--realtime" in argv
        if app is not None:
            app.jobs = jobs
        for src in inputs:
            if sources.is_stream(src):
                cam = next((c for c in db.cameras() if c["url"] == src), None)
                cam_id = cam["id"] if cam else db.add_camera(sources.display_name(src), src)
                jobs.start_camera(db.camera(cam_id))
            else:
                jobs.submit(src, sources.display_name(src), "file")
        if "--test" in argv:
            jobs.start_test_loop(sources.testset())
            print("Test mode: playing the test videos in a loop (Ctrl+C to stop)")
        print("Ready.")

    try:
        if web:
            while True:
                time.sleep(1)
        elif jobs is not None:
            jobs.wait_idle()
            if jobs.cameras or jobs.feeds:
                print("Cameras are running, press Ctrl+C to stop.")
                while True:
                    time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        if jobs is not None:
            jobs.stop_all()
            deadline = time.time() + 10           # let running jobs hand over their last tracks
            while jobs.feeds and time.time() < deadline:
                time.sleep(0.1)
        if recorder is not None:
            recorder.close()
        db.close()


if __name__ == "__main__":
    main()
