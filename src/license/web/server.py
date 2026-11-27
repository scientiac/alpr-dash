"""Tiny dashboard server (standard library only).

Routes:
  /             the dashboard page
  /stream.mjpg  live annotated video (MJPEG)
  /api/state    JSON with the current source, speed, live tracks, totals and
                the most recent records from the database (?q=... to search them)
  /images/...   saved plate and vehicle photos
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

INDEX = (Path(__file__).parent / "index.html").read_bytes()
RECENT = 200        # records sent to the page


class LiveState:
    """Shared between the processing thread and the web server."""

    def __init__(self, db):
        self.db = db
        self.cond = threading.Condition()
        self.jpeg = None
        self.frame_no = 0
        self.info = {"source": "", "fps": 0.0, "active": [], "done": False}

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

    def to_json(self, query=""):
        with self.cond:
            info = dict(self.info)
        info["plates"] = self.db.recent(RECENT, query)
        info["stats"] = self.db.stats()
        return json.dumps(info).encode()


def make_handler(state: LiveState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):   # keep the console quiet
            pass

        def _send(self, body, ctype, cache="no-store"):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlsplit(self.path)
            path = url.path
            if path == "/":
                self._send(INDEX, "text/html; charset=utf-8")
            elif path == "/api/state":
                query = parse_qs(url.query).get("q", [""])[0][:200]
                self._send(state.to_json(query), "application/json")
            elif path == "/stream.mjpg":
                self._stream()
            elif path.startswith("/images/"):
                self._image(unquote(path[len("/images/"):]))
            else:
                self.send_error(404)

        def _image(self, rel):
            root = state.db.image_dir.resolve()
            file = (root / rel).resolve()
            if file.suffix != ".jpg" or root not in file.parents or not file.is_file():
                self.send_error(404)
                return
            # Every saved photo gets a new name, so it can be cached for good
            self._send(file.read_bytes(), "image/jpeg", "max-age=31536000, immutable")

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seen = -1
            try:
                while True:
                    with state.cond:
                        state.cond.wait_for(
                            lambda: state.frame_no != seen or state.info["done"],
                            timeout=1.0)
                        if state.frame_no == seen:
                            continue
                        jpeg, seen = state.jpeg, state.frame_no
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                     b"Content-Length: " + str(len(jpeg)).encode()
                                     + b"\r\n\r\n" + jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def serve(state: LiveState, host: str, port: int):
    """Start the server in a background thread and return it."""
    httpd = ThreadingHTTPServer((host, port), make_handler(state))
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"Dashboard: http://{host}:{port}")
    return httpd
