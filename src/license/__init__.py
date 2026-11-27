"""Real-time license plate recognition with a web dashboard.

Usage:
  license                 process every video in src/testset
  license <file|url>      process one video file or stream
Options:
  --fast                  do not slow video files down to real-time speed
  --no-web                run without the dashboard (console + database only)
  --port N                dashboard port (default 8000)
"""

import sys
import time

from . import config, sources
from .db import Database
from .pipeline import process


def main() -> None:
    argv = sys.argv[1:]
    if "-h" in argv or "--help" in argv:
        print(__doc__)
        return
    port = config.PORT
    if "--port" in argv:
        i = argv.index("--port")
        port = int(argv[i + 1])
        del argv[i:i + 2]
    args = [a for a in argv if not a.startswith("--")]
    web = "--no-web" not in argv
    realtime = "--fast" not in argv

    jobs = sources.resolve(args)
    db = Database()
    print(f"Recording to {config.DB_FILE} (photos in {config.IMAGE_DIR})")

    state = None
    if web:
        from .web.server import LiveState, serve
        state = LiveState(db)
        serve(state, config.HOST, port)

    print("Loading models...")
    from .recognizer import Recognizer
    from .recorder import Recorder
    from .vehicles import VehicleDescriber
    recognizer = Recognizer()
    recorder = Recorder(db, VehicleDescriber())

    try:
        for src in jobs:
            process(src, recognizer, recorder, state, realtime)
        recorder.close()          # finish the last records before saying done
        if state is not None:
            state.finish()
            print("Done. Dashboard stays open, press Ctrl+C to quit.")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        recorder.close()
        db.close()


if __name__ == "__main__":
    main()
