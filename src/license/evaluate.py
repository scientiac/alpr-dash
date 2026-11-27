"""Measure the system on the test videos against manually verified plates.

Usage:  uv run license-eval [ground_truth.csv] [--out DIR]

Every video named in the ground-truth file is processed into a separate,
fresh data folder (default data/eval), so the normal records are untouched.
The results are printed and saved as results.json and results.md.

Ground truth (CSV): video, plate, vehicle_type, difficulty, note
  - one row per vehicle whose plate can be read by a person in the video;
  - difficulty is "clear" or "difficult" (blur, leaves, far away, angle).

Measured:
  - vehicle tracking: share of these vehicles that got a vehicle track with
    the right type;
  - plate detection: share of these vehicles with a plate found (any status);
  - exact match: recognized plate text equal to the ground truth, overall,
    for clear and for difficult plates;
  - false plates: recognized/review records whose text matches no vehicle;
  - duplicates: extra records of the same vehicle;
  - speed: processing frames per second and time per video.
"""

import csv
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

from . import config
from .db import plate_key


def load_truth(path):
    rows = list(csv.DictReader(open(path, newline="")))
    for r in rows:
        r["plate"] = r["plate"].strip().upper()
    return rows


def main():
    argv = sys.argv[1:]
    out = Path("data/eval")
    if "--out" in argv:
        i = argv.index("--out")
        out = Path(argv[i + 1])
        del argv[i:i + 2]
    truth_file = Path(argv[0]) if argv else config.TESTSET_DIR / "ground_truth.csv"
    truth = load_truth(truth_file)
    videos = sorted({r["video"] for r in truth})

    # A fresh data folder just for this evaluation
    shutil.rmtree(out, ignore_errors=True)
    config.DATA_DIR, config.DB_FILE = out, out / "plates.db"
    config.IMAGE_DIR, config.UPLOAD_DIR, config.VIDEO_DIR = out / "images", out / "uploads", out / "videos"
    from . import load_models
    from .db import Database
    from .pipeline import process
    from .recorder import Recorder

    db = Database(config.DB_FILE)
    print("Loading models...")
    models = load_models()
    recorder = Recorder(db, models.describer, models.vehicles)

    speed = {}
    for v in videos:
        jid = db.add_job("file", v, config.TESTSET_DIR / v)
        t = time.perf_counter()
        r = process(str(config.TESTSET_DIR / v), models, recorder, job_id=jid,
                    output=db.video_dir / f"{jid}.webm")
        recorder.wait(jid)
        speed[v] = {"frames": r["frames"], "seconds": round(time.perf_counter() - t, 1),
                    "fps": round(r["fps"], 1), "video_seconds": round(r["duration"], 1)}
    recorder.close()

    records, _ = db.records({}, limit=100000)
    by_video = defaultdict(list)
    for rec in records:
        by_video[rec["source"]].append(rec)

    rows, results = [], {"videos": len(videos), "vehicles": len(truth)}
    tracked = typed = found = exact = 0
    exact_by = defaultdict(lambda: [0, 0])
    used = set()
    for t in truth:
        recs = by_video[t["video"]]
        key = plate_key(t["plate"])
        # The record for this vehicle: its plate, or (if unread) the best guess
        mine = [r for r in recs if r["plate_key"] == key]
        hit = mine[0] if mine else None
        ok = bool(hit) and hit["status"] in ("recognized", "review") and hit["plate"] == t["plate"]
        plate_seen = bool(hit and hit["plate_image"])
        if not hit:   # unreadable records with a plate photo may belong to this vehicle
            cands = [r for r in recs if r["status"] == "unrecognized" and r["plate_image"] and r["id"] not in used]
            if cands:
                plate_seen = True
                hit = cands[0]
        if hit:
            used.add(hit["id"])
        tracked += bool(hit and hit["track_id"])
        typed += bool(hit and hit["vehicle_type"] == t["vehicle_type"])
        found += plate_seen
        exact += ok
        exact_by[t["difficulty"]][0] += ok
        exact_by[t["difficulty"]][1] += 1
        rows.append({"video": t["video"], "truth": t["plate"], "difficulty": t["difficulty"],
                     "read": hit["plate"] if hit else "", "status": hit["status"] if hit else "missed",
                     "confidence": hit["confidence"] if hit else None, "exact": ok,
                     "copies": len(mine), "type": hit["vehicle_type"] if hit else None})

    keys = {plate_key(t["plate"]) for t in truth}
    plated = [r for r in records if r["status"] in ("recognized", "review")]
    false_plates = [r for r in plated if r["plate_key"] not in keys]
    duplicates = sum(max(0, row["copies"] - 1) for row in rows)
    plateless = [r for r in records if not r["plate_image"]]
    frames = sum(s["frames"] for s in speed.values())
    secs = sum(s["seconds"] for s in speed.values())

    results.update({
        "vehicle_tracked": tracked, "vehicle_type_correct": typed,
        "plate_detected": found, "exact_match": exact,
        "exact_clear": exact_by["clear"], "exact_difficult": exact_by["difficult"],
        "records_total": len(records), "records_with_plate_text": len(plated),
        "false_plate_records": [r["plate"] for r in false_plates],
        "duplicate_records": duplicates,
        "duplicate_rate": round(duplicates / max(1, len(plated)), 3),
        "plateless_vehicle_records": len(plateless),
        "status_counts": {s: sum(r["status"] == s for r in records)
                          for s in ("recognized", "review", "unrecognized")},
        "frames": frames, "processing_seconds": round(secs, 1),
        "average_fps": round(frames / max(secs, 1e-6), 1),
        "per_video": speed, "rows": rows,
    })
    (out / "results.json").write_text(json.dumps(results, indent=2))

    n = len(truth)
    pct = lambda a, b: f"{a}/{b} ({100 * a / max(1, b):.0f}%)"
    lines = [
        f"Videos: {len(videos)}, vehicles with a readable plate: {n}",
        f"Vehicle tracked with correct type: {pct(typed, n)}",
        f"Plate detected:                    {pct(found, n)}",
        f"Exact plate match:                 {pct(exact, n)}",
        f"  clear plates:                    {pct(*exact_by['clear'])}",
        f"  difficult plates:                {pct(*exact_by['difficult'])}",
        f"False plate records:               {len(false_plates)} {[r['plate'] for r in false_plates]}",
        f"Duplicate records:                 {duplicates} ({100 * results['duplicate_rate']:.0f}% of plate records)",
        f"Records: {len(records)} total, {results['status_counts']}, "
        f"{len(plateless)} vehicles without a visible plate",
        f"Speed: {frames} frames in {secs:.0f} s = {results['average_fps']} fps",
        "",
        "| Video | Truth | Read | Status | Conf. | Difficulty |",
        "|---|---|---|---|---|---|",
    ] + [f"| {r['video']} | {r['truth']} | {r['read'] or '-'} | {r['status']} | "
         f"{r['confidence'] if r['confidence'] is not None else '-'} | {r['difficulty']} |" for r in rows]
    text = "\n".join(lines)
    (out / "results.md").write_text(text + "\n")
    print(text)
    print(f"\nSaved to {out}/results.json and results.md")
    db.close()


if __name__ == "__main__":
    main()
