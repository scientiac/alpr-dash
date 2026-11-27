"""Dashboard server and JSON API (standard library only).

Pages and files
  /                      the dashboard (index.html, app.js, app.css)
  /stream/<job>.mjpg     live annotated video of a running job (MJPEG)
  /images/<path>         saved plate and vehicle photos
  /videos/<file>         processed (annotated) videos, with seeking

API (JSON)
  GET    /api/state                     running jobs, totals, newest record id, model status
  GET    /api/config                    upload limits
  GET    /api/records?<filters>         records: {total, records}
         filters: q, type, status, direction, watch, job, source, since, until,
                  sort=newest|oldest, limit, offset
  GET    /api/records/<id>              one record
  POST   /api/records/<id>              manual review: {"plate": "...", "status": "..."}
  DELETE /api/records/<id>              delete a record and its photos
  GET    /api/export.csv?<filters>      the filtered records as CSV
  GET    /api/stats?<filters>           counts by status, type, direction, watchlist
  GET    /api/trend?<filters>           records over time by vehicle type
  GET    /api/sources                   source names, for the filter
  POST   /api/upload?name=<file name>   upload a video (the request body is the file)
  GET    /api/jobs                      processing jobs
  POST   /api/jobs/<id>/cancel          stop a queued or running job
  DELETE /api/jobs/<id>                 remove a job and its videos (records stay)
  GET    /api/cameras                   cameras and whether they are running
  POST   /api/cameras                   add {"name": "...", "url": "rtsp://... or 0"}
  DELETE /api/cameras/<id>
  POST   /api/cameras/<id>/start|stop
  GET    /api/watchlist                 whitelist and blacklist
  POST   /api/watchlist                 add {"plate": "...", "kind": "allow|block", "note": "..."}
  DELETE /api/watchlist/<id>
"""

import json
import re
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from .. import config
from ..db import STATUSES, now
from ..sources import VideoError, probe

WEB = Path(__file__).parent
STATIC = {"/": ("index.html", "text/html; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/app.css": ("app.css", "text/css; charset=utf-8")}
FILTERS = ("q", "type", "status", "direction", "watch", "job", "source", "since", "until", "sort")
MAX_JSON = 64 * 1024


class App:
    """What the handlers need. `jobs` is set once the models have loaded."""

    def __init__(self, db):
        self.db = db
        self.jobs = None
        self.error = None          # message if the models could not be loaded


class HTTPError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _id(text):
    if not text.isdigit():
        raise HTTPError(404, "Not found")
    return int(text)


def make_handler(app: App):
    db = app.db

    class Handler(BaseHTTPRequestHandler):
        server_version = "license"
        sys_version = ""

        def log_message(self, *args):   # keep the console quiet
            pass

        # ---------- plumbing ----------
        def _send(self, body, ctype, code=200, headers=None):
            if isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (headers or {"Cache-Control": "no-store"}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, data, code=200):
            self._send(json.dumps(data), "application/json", code)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_JSON:
                raise HTTPError(413, "Request too large")
            try:
                return json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                raise HTTPError(400, "Invalid JSON")

        def _jobs(self):
            if app.jobs is None:
                raise HTTPError(503, app.error or "The models are still loading, try again shortly")
            return app.jobs

        def _dispatch(self, method):
            url = urlsplit(self.path)
            self.query = {k: v[0] for k, v in parse_qs(url.query).items()}
            path = unquote(url.path)
            try:
                handler = self._route(method, path)
                if handler is None:
                    raise HTTPError(404, "Not found")
                handler()
            except HTTPError as e:
                self._json({"error": str(e)}, e.code)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            self._dispatch("GET")

        def do_HEAD(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

        def do_DELETE(self):
            self._dispatch("DELETE")

        def _filters(self):
            return {k: self.query[k] for k in FILTERS if self.query.get(k)}

        # ---------- routing ----------
        def _route(self, method, path):
            p = path.rstrip("/").split("/")[1:] or [""]
            if method == "GET" and path in STATIC:
                name, ctype = STATIC[path]
                return lambda: self._send((WEB / name).read_bytes(), ctype)
            if method == "GET" and p[0] == "images" and len(p) > 1:
                return lambda: self._file(db.image_dir, "/".join(p[1:]), "image/jpeg", ".jpg")
            if method == "GET" and p[0] == "videos" and len(p) == 2:
                return lambda: self._file(db.video_dir, p[1], "video/webm", ".webm")
            if method == "GET" and p[0] == "stream" and len(p) == 2 and p[1].endswith(".mjpg"):
                return lambda: self._stream(_id(p[1][:-5]))
            if p[0] != "api" or len(p) < 2:
                return None
            r = p[1:]
            routes = {
                ("GET", "state"): self.state,
                ("GET", "config"): lambda: self._json({
                    "max_upload_mb": config.MAX_UPLOAD_MB, "video_exts": sorted(config.VIDEO_EXTS)}),
                ("GET", "records"): self.records,
                ("GET", "export.csv"): self.export,
                ("GET", "stats"): lambda: self._json(db.stats(self._filters())),
                ("GET", "trend"): lambda: self._json(db.trend(self._filters())),
                ("GET", "sources"): lambda: self._json(db.sources()),
                ("POST", "upload"): self.upload,
                ("GET", "jobs"): lambda: self._json(db.jobs(100)),
                ("GET", "cameras"): self.cameras,
                ("POST", "cameras"): self.add_camera,
                ("GET", "watchlist"): lambda: self._json(db.watchlist()),
                ("POST", "watchlist"): self.add_watch,
            }
            if len(r) == 1:
                return routes.get((method, r[0]))
            key = (method, r[0], r[2] if len(r) == 3 else None)
            item = r[1]
            more = {
                ("GET", "records", None): lambda: self.record(_id(item)),
                ("POST", "records", None): lambda: self.review(_id(item)),
                ("DELETE", "records", None): lambda: self.delete_record(_id(item)),
                ("POST", "jobs", "cancel"): lambda: self._ok(self._jobs().cancel(_id(item))),
                ("DELETE", "jobs", None): lambda: self._ok(self._jobs().delete(_id(item)),
                                                           "Job not found or still running"),
                ("DELETE", "cameras", None): lambda: self.delete_camera(_id(item)),
                ("POST", "cameras", "start"): lambda: self.camera(_id(item), True),
                ("POST", "cameras", "stop"): lambda: self.camera(_id(item), False),
                ("DELETE", "watchlist", None): lambda: self._ok(db.delete_watch(_id(item)) or True),
            }
            return more.get(key)

        def _ok(self, ok, message="Not found"):
            if not ok:
                raise HTTPError(404, message)
            self._json({"ok": True})

        # ---------- handlers ----------
        def state(self):
            running = app.jobs.running() if app.jobs else []
            self._json({"ready": app.jobs is not None, "error": app.error, "running": running,
                        "stats": db.stats(), "latest_id": db.latest_id(), "time": now()})

        def records(self):
            try:
                limit = min(max(int(self.query.get("limit", 50)), 1), 500)
                offset = max(int(self.query.get("offset", 0)), 0)
            except ValueError:
                raise HTTPError(400, "limit and offset must be numbers")
            rows, total = db.records(self._filters(), limit, offset)
            self._json({"total": total, "records": rows})

        def record(self, rid):
            r = db.get(rid)
            if not r:
                raise HTTPError(404, "Record not found")
            self._json(r)

        def review(self, rid):
            if not db.get(rid):
                raise HTTPError(404, "Record not found")
            body, change = self._body(), {}
            if "plate" in body:
                plate = re.sub(r"[^A-Z0-9]", "", str(body["plate"]).upper())
                if len(plate) > 12:
                    raise HTTPError(400, "Plate is too long")
                change["plate"] = plate
            if "status" in body:
                if body["status"] not in STATUSES:
                    raise HTTPError(400, "Unknown status")
                change["status"] = body["status"]
            change["reviewed"] = 1
            db.update(rid, change)
            self._json(db.get(rid))

        def delete_record(self, rid):
            self._ok(db.delete(rid), "Record not found")

        def export(self):
            name = f"traffic-records-{now()[:10]}.csv"
            self._send(db.csv(self._filters()), "text/csv; charset=utf-8", headers={
                "Cache-Control": "no-store", "Content-Disposition": f'attachment; filename="{name}"'})

        def upload(self):
            jobs = self._jobs()
            name = Path(self.query.get("name", "")).name
            safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "video"
            if Path(safe).suffix.lower() not in config.VIDEO_EXTS:
                raise HTTPError(415, f"Unsupported file type. Use one of: "
                                     f"{', '.join(sorted(config.VIDEO_EXTS))}")
            size = int(self.headers.get("Content-Length") or 0)
            if size <= 0:
                raise HTTPError(411, "The file is empty")
            if size > config.MAX_UPLOAD_MB * 1024 * 1024:
                raise HTTPError(413, f"The file is larger than {config.MAX_UPLOAD_MB} MB")
            path = db.upload_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{secrets.token_hex(3)}_{safe}"
            left = size
            try:
                with open(path, "wb") as f:
                    while left:
                        chunk = self.rfile.read(min(left, 1 << 20))
                        if not chunk:
                            raise HTTPError(400, "The upload was interrupted")
                        f.write(chunk)
                        left -= len(chunk)
                probe(path)
            except (HTTPError, VideoError, OSError) as e:
                path.unlink(missing_ok=True)
                msg = str(e)
                if isinstance(e, VideoError) and msg.startswith("Could not open"):
                    msg = "This file could not be opened as a video (corrupted or an unsupported codec)"
                jid = db.add_job("upload", name or safe, path)
                db.update_job(jid, status="failed", finished=now(), error=msg)
                raise HTTPError(e.code if isinstance(e, HTTPError) else 422, msg)
            jid = jobs.submit(path, name or safe)
            self._json(db.job(jid), 201)

        def cameras(self):
            running = app.jobs.cameras if app.jobs else {}
            out = []
            for c in db.cameras():
                jid = running.get(c["id"])
                job = db.job(jid) if jid else None
                state = None
                if job:
                    state = "reconnecting" if job["error"] else "running" if job["frames"] else "connecting"
                out.append({**c, "job": jid, "state": state, "error": job["error"] if job else None})
            self._json(out)

        def add_camera(self):
            body = self._body()
            name, url = str(body.get("name", "")).strip(), str(body.get("url", "")).strip()
            if not name or not url:
                raise HTTPError(400, "A camera needs a name and a URL or device number")
            if not (url.isdigit() or url.lower().startswith(config.STREAM_PREFIXES)):
                raise HTTPError(400, "Use an rtsp://, http(s):// ... URL or a device number such as 0")
            self._json({"id": db.add_camera(name[:80], url[:500])}, 201)

        def delete_camera(self, cid):
            if app.jobs:
                app.jobs.stop_camera(cid)
            db.delete_camera(cid)
            self._json({"ok": True})

        def camera(self, cid, start):
            jobs, cam = self._jobs(), db.camera(cid)
            if not cam:
                raise HTTPError(404, "Camera not found")
            self._json({"job": jobs.start_camera(cam)} if start else {"ok": jobs.stop_camera(cid)})

        def add_watch(self):
            body = self._body()
            plate = re.sub(r"[^A-Z0-9]", "", str(body.get("plate", "")).upper())
            kind = body.get("kind")
            if not 2 <= len(plate) <= 12:
                raise HTTPError(400, "Enter a plate number")
            if kind not in ("allow", "block"):
                raise HTTPError(400, "List must be allow or block")
            db.add_watch(plate, kind, str(body.get("note", ""))[:200])
            self._json({"ok": True}, 201)

        # ---------- files and streams ----------
        def _file(self, root, rel, ctype, suffix):
            root = root.resolve()
            file = (root / rel).resolve()
            if file.suffix != suffix or root not in file.parents or not file.is_file():
                raise HTTPError(404, "Not found")
            size = file.stat().st_size
            start, end = 0, size - 1
            rng = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            if rng and (rng[1] or rng[2]):        # seeking in the video player
                if rng[1]:
                    start, end = int(rng[1]), int(rng[2]) if rng[2] else size - 1
                else:
                    start = max(0, size - int(rng[2]))
                end = min(end, size - 1)
                if start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
            self.send_response(206 if rng and (rng[1] or rng[2]) else 200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if rng and (rng[1] or rng[2]):
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            # Every saved photo gets a new name, so photos can be cached for good
            self.send_header("Cache-Control", "private, max-age=31536000, immutable"
                             if suffix == ".jpg" else "private, no-cache")
            self.end_headers()
            if self.command == "HEAD":
                return
            with open(file, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left:
                    chunk = f.read(min(left, 1 << 20))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)

        def _stream(self, job_id):
            feed = app.jobs.feeds.get(job_id) if app.jobs else None
            if feed is None:
                raise HTTPError(404, "This job is not running")
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seen = -1
            while True:
                with feed.cond:
                    feed.cond.wait_for(lambda: feed.frame_no != seen or feed.info["done"], timeout=1.0)
                    # A camera that is reconnecting keeps its feed: wait for new frames
                    if feed.info["done"] and feed.frame_no == seen and app.jobs.feeds.get(job_id) is not feed:
                        return
                    if feed.frame_no == seen or feed.jpeg is None:
                        continue
                    jpeg, seen = feed.jpeg, feed.frame_no
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")

    return Handler


def serve(app: App, host: str, port: int):
    """Start the server in a background thread and return it."""
    try:
        httpd = ThreadingHTTPServer((host, port), make_handler(app))
    except OSError as e:
        raise SystemExit(f"Cannot start the dashboard on {host}:{port} ({e.strerror}). "
                         f"Is it already running? Use --port to pick another port.")
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"Dashboard: http://{host}:{port}")
    return httpd
