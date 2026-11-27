"""Processing jobs: uploaded or given video files, and live cameras.

Files are processed one at a time, in the order they arrive, by one worker
thread (each already uses several CPU threads). Every camera runs in its own
thread for as long as it is switched on, and reconnects if the stream drops.
Each running job has a Feed for the live view and a stop switch.
"""

import queue
import threading
import time
import traceback
from pathlib import Path

from . import config
from .db import now
from .pipeline import Feed, process
from .sources import VideoError

RECONNECT_S = 5


class JobManager:
    def __init__(self, db, models, recorder):
        self.db, self.models, self.recorder = db, models, recorder
        self.feeds: dict[int, Feed] = {}          # running job -> live view
        self._stops: dict[int, threading.Event] = {}
        self.cameras: dict[int, int] = {}         # camera id -> its running job
        self._queue = queue.Queue()
        self.realtime = False                     # play files at natural speed
        self._resume()
        threading.Thread(target=self._worker, daemon=True).start()

    # ---------- files ----------
    def submit(self, path, name, kind="upload"):
        job_id = self.db.add_job(kind, name, path)
        self._queue.put(job_id)
        return job_id

    def _resume(self):
        """Jobs cut off by the last shutdown: queued ones run again,
        ones that were half done are marked as interrupted."""
        for j in self.db.interrupted_jobs():
            if j["kind"] in ("camera", "test") or j["status"] == "processing":
                self.db.update_job(j["id"], status="failed", finished=now(),
                                   error="Interrupted when the program stopped")
            elif Path(j["path"]).is_file():
                self._queue.put(j["id"])

    def _worker(self):
        while True:
            job_id = self._queue.get()
            job = self.db.job(job_id)
            if job and job["status"] == "queued":
                self._run_file(job)

    def _run_file(self, job):
        jid = job["id"]
        stop, feed = threading.Event(), Feed()
        self._stops[jid], self.feeds[jid] = stop, feed
        output = (self.db.video_dir / f"{jid}.webm") if config.SAVE_ANNOTATED else None
        self.db.update_job(jid, status="processing", started=now(), progress=0, error=None)

        def progress(fraction, frames, fps):
            self.db.update_job(jid, progress=round(fraction, 4), frames=frames, fps=round(fps, 1))

        try:
            r = process(job["path"], self.models, self.recorder, name=job["name"], job_id=jid, feed=feed,
                        realtime=self.realtime, stop=stop, progress=progress, output=output)
            self.recorder.wait(jid)            # "done" means the records are stored
            note = None
            if r["cut_short"]:
                note = (f"Stopped at frame {r['frames']} of {r['total']}: "
                        "the rest of the file could not be decoded")
            if not stop.is_set():
                self.db.update_job(jid, progress=1)
            self.db.update_job(jid, status="stopped" if stop.is_set() else "done", finished=now(),
                               frames=r["frames"], fps=round(r["fps"], 1),
                               duration=round(r["duration"], 2), error=note,
                               output=output.name if r["annotated"] else None)
        except VideoError as e:
            self.db.update_job(jid, status="failed", finished=now(), error=str(e))
        except Exception as e:                 # a broken job must not stop the worker
            traceback.print_exc()
            self.db.update_job(jid, status="failed", finished=now(), error=f"Processing error: {e}")
        finally:
            self._stops.pop(jid, None)
            self.feeds.pop(jid, None)

    def cancel(self, job_id):
        job = self.db.job(job_id)
        if not job:
            return False
        if job_id in self._stops:
            self._stops[job_id].set()
        elif job["status"] == "queued":
            self.db.update_job(job_id, status="stopped", finished=now())
        return True

    def delete(self, job_id):
        """Remove a finished job, its uploaded video and annotated video (records stay)."""
        job = self.db.job(job_id)
        if not job or job_id in self._stops:
            return False
        if job["output"]:
            (self.db.video_dir / job["output"]).unlink(missing_ok=True)
        path = Path(job["path"])
        if job["kind"] == "upload" and path.parent.resolve() == self.db.upload_dir.resolve():
            path.unlink(missing_ok=True)
        self.db.delete_job(job_id)
        return True

    # ---------- cameras ----------
    def start_camera(self, cam):
        if cam["id"] in self.cameras:
            return self.cameras[cam["id"]]
        jid = self.db.add_job("camera", cam["name"], cam["url"])
        stop, feed = threading.Event(), Feed()
        self._stops[jid], self.feeds[jid], self.cameras[cam["id"]] = stop, feed, jid
        threading.Thread(target=self._run_camera, args=(cam, jid, stop, feed), daemon=True).start()
        return jid

    def _run_camera(self, cam, jid, stop, feed):
        self.db.update_job(jid, status="processing", started=now())
        frames = 0
        try:
            while not stop.is_set():
                try:
                    r = process(cam["url"], self.models, self.recorder, name=cam["name"], job_id=jid, feed=feed,
                                stop=stop, progress=lambda f, n, fps: self.db.update_job(
                                    jid, frames=frames + n, fps=round(fps, 1), error=None))
                    frames += r["frames"]
                    error = "Stream ended; reconnecting"
                except VideoError as e:
                    error = f"{e}; retrying every {RECONNECT_S} s"
                if stop.is_set():
                    break
                self.db.update_job(jid, error=error)
                feed.info["done"] = False
                stop.wait(RECONNECT_S)
            self.db.update_job(jid, status="stopped", finished=now(), error=None)
        except Exception as e:
            traceback.print_exc()
            self.db.update_job(jid, status="failed", finished=now(), error=f"Processing error: {e}")
        finally:
            self._stops.pop(jid, None)
            self.feeds.pop(jid, None)
            self.cameras.pop(cam["id"], None)

    # ---------- test mode ----------
    def start_test_loop(self, videos):
        """Play the test videos one after another, at their natural speed,
        over and over, like a live camera (for demonstrations)."""
        jid = self.db.add_job("test", "Test videos (loop)", f"{len(videos)} videos")
        stop, feed = threading.Event(), Feed()
        self._stops[jid], self.feeds[jid] = stop, feed

        def run():
            self.db.update_job(jid, status="processing", started=now())
            frames, rounds = 0, 0
            try:
                while not stop.is_set():
                    rounds += 1
                    for path in videos:
                        if stop.is_set():
                            break
                        name = Path(path).name
                        self.db.update_job(jid, error=f"Round {rounds}: {name}")
                        try:
                            r = process(path, self.models, self.recorder, name=name, job_id=jid,
                                        feed=feed, realtime=True, stop=stop,
                                        progress=lambda f, n, fps: self.db.update_job(
                                            jid, frames=frames + n, fps=round(fps, 1)))
                            frames += r["frames"]
                        except VideoError as e:
                            print(f"Skipping {name}: {e}")
                        with feed.cond:
                            feed.info["done"] = False
                self.db.update_job(jid, status="stopped", finished=now(), error=None)
            except Exception as e:
                traceback.print_exc()
                self.db.update_job(jid, status="failed", finished=now(), error=f"Processing error: {e}")
            finally:
                self._stops.pop(jid, None)
                self.feeds.pop(jid, None)

        threading.Thread(target=run, daemon=True).start()
        return jid

    def stop_camera(self, cam_id):
        jid = self.cameras.get(cam_id)
        if jid is not None and jid in self._stops:
            self._stops[jid].set()
        return jid is not None

    def stop_all(self):
        for ev in list(self._stops.values()):
            ev.set()

    def wait_idle(self):
        """Block until every queued file has been processed (for the command line)."""
        while True:
            busy = [j for j in self.db.jobs(1000)
                    if j["status"] in ("queued", "processing") and j["kind"] != "camera"]
            if not busy:
                return
            time.sleep(0.5)

    def running(self):
        """Jobs being processed right now, with their live numbers."""
        out = []
        for jid, feed in list(self.feeds.items()):
            job = self.db.job(jid)
            if job:
                with feed.cond:
                    job["live"] = dict(feed.info)
                out.append(job)
        return out
