"""Tiny dashboard server (standard library only).

Routes:
  /             the dashboard page
  /stream.mjpg  live annotated video (MJPEG)
  /api/state    JSON with the current source, speed, live tracks and logged plates
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

INDEX = (Path(__file__).parent / "index.html").read_bytes()


class LiveState:
    """Shared between the processing thread and the web server."""

    def __init__(self, store):
        self.store = store
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

    def to_json(self):
        with self.cond:
            info = dict(self.info)
        info["plates"] = self.store.snapshot()
        return json.dumps(info).encode()


def make_handler(state: LiveState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):   # keep the console quiet
            pass

        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                self._send(INDEX, "text/html; charset=utf-8")
            elif path == "/api/state":
                self._send(state.to_json(), "application/json")
            elif path == "/stream.mjpg":
                self._stream()
            else:
                self.send_error(404)

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
