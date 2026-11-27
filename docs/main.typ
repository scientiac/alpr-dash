#import "@preview/charged-ieee:0.1.4": ieee

#show link: it => {
  set text(blue)
  if type(it.dest) != str {
    it
  }
  else {
    underline(it)
  }
}

#show raw.where(block: true): set text(size: 7pt)

#show: ieee.with(
  title: [AI-Based Automatic Vehicle Number Plate Detection and Recognition System],
  abstract: [
    This project turns a traffic video into searchable vehicle records. A user uploads a video in a web dashboard (or connects a live camera); the system detects and tracks every vehicle, finds and reads its number plate, and stores one record per vehicle with the date and time, the time in the video, the plate text and confidence, the vehicle type and tracking ID, a recognition status, and photos of the plate and the vehicle.

    Four small neural networks run locally on an ordinary CPU. Vehicles are tracked with ByteTrack; each plate is read many times and the readings are combined by a vote, and uncertain results are marked for review instead of being trusted. Records are kept in SQLite. The dashboard has an overview, video upload with progress and annotated playback, a searchable and filterable records table with CSV export, analytics, live cameras and a whitelist/blacklist. On 14 test videos all 15 readable plates were read exactly, with no false or duplicate plate records, at about 9 frames per second.
  ],
  authors: (
    (
      name: "Spandan Guragain",
      email: "spandan@3o14.com"
    ),
  ),
  index-terms: ("Number plate recognition", "OCR", "YOLO", "ByteTrack", "SQLite", "Web dashboard"),
  figure-supplement: [Fig.],
)

#outline()

= Introduction

Automatic number plate recognition (ANPR) is used for parking, access control and traffic monitoring. The system has to find vehicles and their small plates in a busy picture, read the characters despite blur, shadows and odd angles, record each vehicle once, and say honestly when it is not sure.

The first prototype of this project used classical image processing (edges, thresholds, contour shapes) with the general-purpose Tesseract OCR engine. It needed many hand-tuned numbers, confused similar characters and mistook grilles and leaves for plates. It was replaced by models trained for the job, which is simpler and far more accurate. This report describes the complete system built to the internship assignment: upload, detection, tracking, recognition, database, dashboard, and the second-phase features except user accounts.

== What the system does

+ The user uploads a traffic video (MP4, AVI, MOV, MKV, WebM) in the dashboard; it is checked and queued.
+ Each frame: vehicles are detected and tracked (ByteTrack), plates are detected separately, linked to their vehicle, cropped and read (OCR).
+ When a vehicle leaves, one record is stored with its photos; uncertain readings are marked "requires review".
+ The dashboard shows progress, the annotated video, the records (search, filters, sort, CSV export), statistics and charts.
+ Second phase: live cameras (RTSP/HTTP streams and webcams) and a whitelist/blacklist with alerts.

= Requirements Coverage

@tab:req maps each requirement of the assignment to what was built.

#figure(
  table(
    columns: (auto, 1fr),
    align: (left, left),
    stroke: 0.5pt,
    inset: 4pt,
    [*Requirement*], [*Implementation*],
    [A. Upload], [Drag-and-drop or file picker; type and size checked in the browser and on the server; upload progress; every file is test-decoded, corrupted ones fail with a message; queue with status and progress; annotated video playable in the browser.],
    [B. Vehicles], [RF-DETR detects cars, motorcycles, buses, trucks (vans via CLIP); boxes and tracking IDs drawn; ByteTrack; duplicate merging.],
    [C. Plates], [Separate plate detector; crop; contrast enhancement and sharpening when a reading is unsure; OCR; confidence; voting over many frames; status recognized / requires review / unrecognized.],
    [D. Logging], [Every field listed (ID, date/time, video time, plate, type, tracking ID, confidence, plate image, vehicle image, file name, status) in SQLite; kept across restarts.],
    [E. Dashboard], [Overview, Upload and Processing results, Records, Analytics, CSV export; plus Live and Watchlist.],
    [3. Table], [All eight columns; search by full or partial plate; filters by type, status, source, date/time range; newest/oldest; evidence photos open full size; CSV of the filtered records.],
    [6. Testing], [`license-eval` with manually verified plates (@sec:results).],
    [Phase 2], [Live CCTV/IP cameras and webcams with reconnecting, live annotated view, whitelist/blacklist with alerts. User accounts were left out on purpose.],
  ),
  caption: [Requirements and how they are met.],
) <tab:req>

= Architecture

The code is split by job, and the AI processing is separate from the web code (@tab:files).

#figure(
  table(
    columns: (auto, 1fr),
    align: (left, left),
    stroke: 0.5pt,
    inset: 4pt,
    [*File*], [*Purpose*],
    [`config.py`], [All settings],
    [`sources.py`], [Files, streams, cameras; checking videos],
    [`recognizer.py`], [Plate detection, OCR, enhancement],
    [`vehicles.py`], [Vehicle detection; colour, make, van],
    [`bytetrack.py`], [Vehicle tracking (ByteTrack)],
    [`tracker.py`], [Plate tracking, voting, status, direction],
    [`pipeline.py`], [The per-frame loop, drawing, annotated video],
    [`recorder.py`], [Background recording and duplicate merging],
    [`jobs.py`], [Upload queue, cameras, progress, cancelling],
    [`db.py`], [SQLite tables, search, statistics, CSV],
    [`evaluate.py`], [Accuracy and speed measurement],
    [`web/server.py`], [HTTP server and JSON API],
    [`web/index.html`, `app.js`, `app.css`], [Dashboard],
    [`__init__.py`], [Command line],
  ),
  caption: [Source files in `src/license`.],
) <tab:files>

Three kinds of threads work together. The *web server* answers the browser. One *job worker* processes uploaded videos one after another (each already uses several CPU cores), and each running *camera* has its own thread. A single *recorder* thread receives the vehicles that left the picture and does the slower work: describing the vehicle, merging duplicates, saving photos and writing the database. This keeps the video loop fast.

= Models

Four neural networks are used (@tab:models). All are ONNX files that run with ONNX Runtime on the CPU; they are downloaded on the first run and cached, after which the system works offline.

#figure(
  table(
    columns: (auto, 1fr, auto),
    align: (left, left, right),
    stroke: 0.5pt,
    inset: 4pt,
    [*Job*], [*Model and licence*], [*Size*],
    [Find plates], [YOLOv9-t 512 px plate detector, `open-image-models` (MIT)], [8 MB],
    [Read plates], [CCT-S v2 global plate OCR, `fast-plate-ocr` (MIT)], [5 MB],
    [Find vehicles, type], [RF-DETR nano 384 px, COCO (Apache-2.0)], [115 MB],
    [Colour, make, van], [MobileCLIP-B (LT), 8-bit (Apple research licence)], [150 MB],
  ),
  caption: [Models used.],
) <tab:models>

*Licences.* The plate detector and OCR packages are MIT licensed; the plate detector is based on the YOLOv9 design, whose original code is GPL-3.0, which should be checked before commercial use. RF-DETR is Apache-2.0. The MobileCLIP weights are licensed by Apple for *non-commercial research only*; they are used only for colour, make and van, and `DESCRIBE_VEHICLES = False` in `config.py` turns them off (vehicle type still works). For a commercial product this model must be replaced.

== Choices against the suggested stack

The assignment suggests a stack; @tab:stack shows where this project differs and why.

#figure(
  table(
    columns: (auto, auto, 1fr),
    align: (left, left, left),
    stroke: 0.5pt,
    inset: 4pt,
    [*Part*], [*Suggested*], [*Used, and why*],
    [Vehicles], [YOLO], [RF-DETR nano: already part of the installed package, Apache-2.0 (Ultralytics YOLO is AGPL), no PyTorch needed.],
    [OCR], [PaddleOCR / EasyOCR], [A plate-specific OCR model: 5 MB, about 3 ms per plate, trained on plates; the general OCR engines are much larger and slower.],
    [Backend], [Flask / FastAPI], [Python's built-in HTTP server: no extra dependency, enough for one machine.],
    [Charts], [Chart.js], [Small hand-written SVG charts: works offline, no library.],
    [Tracking, DB, video], [ByteTrack, SQLite, OpenCV], [As suggested.],
  ),
  caption: [Suggested and used technology.],
) <tab:stack>

= Processing a Video

Every frame passes through the steps in @fig:pipeline.

#figure(
  table(
    columns: (auto, 1fr),
    align: (center, left),
    stroke: 0.5pt,
    inset: 4pt,
    [*\#*], [*Step*],
    [1], [Read a frame],
    [2], [Every 3rd frame: detect vehicles; every frame: ByteTrack predicts and updates the vehicle tracks],
    [3], [Detect plates; follow each plate with the plate tracker],
    [4], [Link each plate to the vehicle box that contains it],
    [5], [Read plates (OCR), retry on an enhanced crop when unsure; vote],
    [6], [Tracks that ended go to the recorder (background)],
    [7], [Draw boxes, IDs and readings; write the annotated video; update the live view and progress],
  ),
  caption: [Processing steps for each frame.],
) <fig:pipeline>

== Vehicle detection and tracking

RF-DETR finds every car, motorcycle, bus and truck in the frame. It takes about 100 ms, so it runs on every third frame of a video (every second frame of a live stream, which already skips frames).

ByteTrack gives each vehicle a stable *tracking ID*. A Kalman filter predicts where each vehicle will be in the next frame, so the boxes move smoothly on the frames without detection. Detections are matched to the predictions by box overlap in two rounds: confident detections first, then weak detections (often a vehicle that is partly hidden) to the tracks still unmatched, which keeps a hidden vehicle's ID instead of starting a new one. New tracks start only from confident detections, and a lost track is kept for 30 frames before it ends.

COCO has no van class. When a record is made, CLIP decides between "car/truck" and "van" and renames the type when it is at least 70% sure.

== Plate detection and reading

The plate detector is a separate model, as the assignment requires. Its boxes are followed across frames by a simple tracker (box overlap or close centres), and each plate is linked to the vehicle box that contains its centre.

Each plate is cut out with a small margin and read by the OCR model, which returns the characters and how sure it is about each. *Preprocessing where necessary*: if the model is less than 80% sure (blur, shadow, a small or angled plate), the crop is enlarged if it is small, its contrast is evened out (CLAHE on brightness only) and it is sharpened, then read again; the better reading is kept. Clear plates are not changed.

A single frame can be blurred, or the car still far away. The system keeps the 12 most confident readings of each plate and votes: the most common text length wins, then for each position the character with the highest total confidence. For example `KR154K`, `KR154GK`, `KR154GK` give `KR154GK`. The *agreement* is the share of characters, over all readings, that match the voted text.

== Recognition status

Instead of trusting every reading, each record gets a status:

- *Recognized*: read at least 4 times, voted confidence at least 90% and agreement at least 85%.
- *Requires review*: a plausible plate (at least 2 readings, 4--10 characters, agreement at least 60%) that misses one of those limits. A person should check it in the dashboard, where the text can be corrected and the record marked as checked.
- *Unrecognized*: the plate was detected at least 8 times but never read reliably (its photo is kept), or the vehicle passed without its plate ever being seen.

Vehicles without a plate in view are recorded only if they move (at least 8% of the frame, so parked cars are skipped) and come close (at least 10% of the frame area, so cars on a far road are skipped).

== Duplicates

Consecutive frames of one vehicle must give one record. The trackers already join frames into one track per vehicle and plate; the recorder then merges records of one run that belong together:

- the same vehicle tracking ID (a plate and its vehicle);
- the same plate text within 30 seconds (a plate track broken by an obstacle);
- the same type and colour and either mostly the same plate text (e.g. a far-away partial reading `01KN` and the full `KR701KN`), or one plate inside the other's vehicle box at the same moment;
- the same type and colour, and a vehicle track that starts within 3 seconds near where the other ended (hidden for a moment).

The best record is kept: recognized before review before unrecognized, then the more confident reading.

== Direction

The first and last boxes of a track are compared: a plate or vehicle that grows by 15% or moves down the picture by 5% comes towards the camera ("in"); one that shrinks or moves up goes away ("out"); otherwise the direction is left empty. `TOWARDS_CAMERA` in `config.py` swaps them for a camera facing the exit.

= Storage

Everything is stored in the `data` folder, which is created readable by the current user only.

== Database

SQLite (part of Python, one file, no server) keeps four tables: `sightings` (the traffic records, @tab:schema), `jobs` (uploads and cameras with status, progress, speed and errors), `cameras` and `watchlist`. Write-ahead logging lets the dashboard read while the recorder writes, and the data survives restarts. Databases from earlier versions are upgraded automatically on start.

#figure(
  table(
    columns: (auto, 1fr),
    align: (left, left),
    stroke: 0.5pt,
    inset: 4pt,
    [*Column*], [*Meaning*],
    [`id`], [Record ID],
    [`time`], [Date and time the record was made],
    [`video_time`], [Seconds from the start of the video],
    [`source`, `job_id`], [Video file name or camera; its job],
    [`plate`], [Recognised plate ('' if unreadable)],
    [`vehicle_type`], [car, van, motorcycle, bus, truck],
    [`track_id`], [Vehicle tracking ID],
    [`confidence`, `readings`], [OCR confidence 0--100; readings voted],
    [`status`, `reviewed`], [recognized / review / unrecognized; checked by a person],
    [`plate_image`, `vehicle_image`], [Photo paths],
    [`direction`, `colour`, `make`], [Extra vehicle details],
  ),
  caption: [Main columns of the `sightings` table.],
) <tab:schema>

== Photos and videos

Photos are JPEG files in `data/images`, one folder per day; the database stores their path. Files keep the database small, are easy to back up, and are sent straight to the browser. Uploaded videos are kept in `data/uploads` and the annotated videos (WebM, which browsers play) in `data/videos`.

== API

The browser uses a small JSON API, which other programs can use too (@tab:api).

#figure(
  table(
    columns: (auto, 1fr),
    align: (left, left),
    stroke: 0.5pt,
    inset: 4pt,
    [*Endpoint*], [*Purpose*],
    [`POST /api/upload?name=`], [Upload a video (body = file)],
    [`GET /api/jobs`], [Processing status and progress],
    [`POST /api/jobs/<id>/cancel`], [Stop a job],
    [`GET /api/records?…`], [Search, filter, sort, page],
    [`GET/POST/DELETE /api/records/<id>`], [Get, review, delete a record],
    [`GET /api/export.csv?…`], [Filtered records as CSV],
    [`GET /api/stats`, `/api/trend`], [Statistics, traffic over time],
    [`/api/cameras`, `/api/watchlist`], [Manage cameras and lists],
    [`/stream/<job>.mjpg`, `/videos/…`], [Live and annotated video],
  ),
  caption: [Main API endpoints.],
) <tab:api>

= Dashboard

The dashboard (`http://127.0.0.1:8000`) is one page with a minimal look and light and dark themes (following the system, or chosen with the button in the header). Its sections follow the way the app is used: see what is happening, watch it, feed it videos, then work with the records. The header with the section tabs stays at the top; the tabs jump to a section and follow the scroll, and each section's title (with the record filters) stays visible while scrolling through it. The sections are:

- *Overview*: total records, recognized, requiring review and unreadable plates, vehicle types, incoming/outgoing and what is being processed now.
- *Upload and processing*: drop or choose videos; upload progress; a table of processing results with status, progress bar, speed, length and record count, and buttons to watch the annotated video, open its records, cancel or remove it. Invalid or corrupted files are shown as failed with the reason.
- *Traffic records* (@fig:records): the table with all eight required columns; search by full or partial plate (also colour, make, type, direction); filters for vehicle type, status, source and a date/time range; newest or oldest first; pages of 50; *Export CSV* of exactly the filtered records. Clicking a row opens the evidence photos (full size in a new tab), all details, and the review tools: correct the plate, change the status, add the plate to a list, or delete the record.
- *Analytics*: vehicles by category, recognition status, direction and watchlist hits, and traffic over time by vehicle type (5-minute, hourly or daily bars depending on the time span), with a table view of the same numbers.
- *Live*: the annotated video of whatever is being processed, the vehicles and plates in view, and the cameras.
- *Watchlist*: the whitelist and blacklist.

#figure(
  image("figures/dashboard_records.jpg", width: 100%),
  caption: [Traffic records with search and filters.],
) <fig:records>

= Second Phase

*Live cameras.* A camera is an RTSP/HTTP stream URL or a webcam number (`0`). Started cameras are processed in real time: a reader thread keeps only the newest frame, so the system never falls behind. A dropped stream is retried every 5 seconds, and the dashboard shows "connecting", "running" or "reconnecting". This was tested with an MJPEG IP-camera stream and with an unreachable address.

*Whitelist and blacklist.* Plates are compared after removing spaces and treating easily confused characters as equal (O/0, I/1, ...). Records show which list a plate is on, the lists are applied to old records too, and a newly recorded blacklisted plate raises an alert in the dashboard (and in the console).

*User accounts* were not built, as agreed. The server therefore listens only on this computer by default (`127.0.0.1`); it must not be exposed to a network without adding authentication.

= Installation

== Requirements

- A 64-bit Linux, macOS or Windows computer; no GPU needed.
- About 1 GB of disk space for packages and models, plus space for data (about 50 kB of photos per vehicle, and the uploaded and annotated videos).
- Internet for the first run only.
- `uv`, which also installs Python 3.12.

No other system packages are needed: the OpenCV package contains FFmpeg for reading and writing video and streams, and SQLite is part of Python. Optional: `sqlite3` to query the database, `typst` to rebuild this document.

== Steps

+ Install `uv`:
  ```
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
  On Windows:
  ```
  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
+ Get the code and install the packages:
  ```
  git clone <repository> license
  cd license
  uv sync
  ```
+ Start the dashboard (the first start downloads the models, about 280 MB):
  ```
  uv run license
  ```
+ Open `http://127.0.0.1:8000`, go to *Upload* and drop a video.
+ Optional: `uv run license --test` plays the test videos in a loop as a demonstration; `uv run license-eval` measures accuracy and speed; `typst compile docs/main.typ` rebuilds this document.

== Dependencies

@tab:deps lists the Python packages used directly (declared in `pyproject.toml`, exact versions pinned in `uv.lock`). The browser side uses no libraries.

#figure(
  table(
    columns: (auto, 1fr),
    align: (left, left),
    stroke: 0.5pt,
    inset: 4pt,
    [*Package*], [*Used for*],
    [`fast-alpr[onnx]`], [Plate detector and OCR; brings `onnxruntime`, `open-image-models` (also the vehicle detector) and `fast-plate-ocr`],
    [`opencv-python-headless`], [Video files and streams, drawing, JPEG and WebM],
    [`numpy`], [Arrays, Kalman filter, voting],
    [`tokenizers`], [Text input for the CLIP model],
    [`huggingface-hub`], [Downloading the CLIP model],
  ),
  caption: [Python packages.],
) <tab:deps>

The standard library covers the rest: `sqlite3` (database), `http.server` (web server), `threading` and `queue` (background work), `csv` (export).

== Usage

#figure(
  table(
    columns: (auto, 1fr),
    align: (left, left),
    stroke: 0.5pt,
    inset: 4pt,
    [*Command*], [*What it does*],
    [`uv run license`], [Start the dashboard],
    [`… license video.mp4`], [Also process a file now],
    [`… license --testset`], [Also process all test videos],
    [`… license rtsp://cam/1`], [Also start a live camera],
    [`… license --test`], [Play all test videos in a loop, like a live camera],
    [`… license --no-web a.mp4`], [Process, print records, exit],
    [`--realtime`], [Play files at natural speed],
    [`--port N`, `--host H`], [Dashboard address],
    [`uv run license-eval`], [Evaluate on the test set],
  ),
  caption: [Command line.],
) <tab:usage>

All settings, including thresholds, the colour and make lists, the direction and the upload size limit (2 GB), are in `config.py`.

= Testing and Evaluation <sec:results>

== Test data and method

The test set is 14 short videos (1280×720, 25 fps, about 20 s each, 7084 frames) recorded at a car park gate, showing Polish (Kraków, `KR`) plates in daylight. The file names (`agh_src*_hrc0`) suggest they come from a video quality test set of AGH University in Kraków; the exact source and its terms of use should be confirmed before the videos are shared further. The ground truth (`src/testset/ground_truth.csv`) lists the 15 vehicles whose plate a person can read in the video. Each plate was read by eye from the video frames and the saved plate photos (@fig:plates); three were marked *difficult* (low resolution, leaves, deep shadow). Other vehicles pass sideways behind the gate without a visible plate; these are not in the ground truth.

`uv run license-eval` processes every video into a separate data folder and compares the records with the ground truth. The figures below are from that run on a 16-thread CPU without a GPU.

== Results

#figure(
  table(
    columns: (1fr, auto),
    align: (left, right),
    stroke: 0.5pt,
    inset: 4pt,
    [*Measure*], [*Result*],
    [Vehicles tracked, correct type], [15 / 15],
    [Plate detected], [15 / 15],
    [Exact plate match, all], [15 / 15 (100%)],
    [#h(1em) clear plates], [12 / 12],
    [#h(1em) difficult plates], [3 / 3],
    [Status "recognized" for these], [15 / 15],
    [False plate records (wrong text)], [0],
    [Duplicate plate records], [0],
    [Records in total], [45],
    [#h(1em) unrecognized (no readable plate)], [30],
    [Average speed (uploaded files)], [8.7 fps],
  ),
  caption: [Evaluation on the 14 test videos.],
) <tab:results>

All 15 plates were read exactly and marked recognized, with no wrong plate text and no duplicate plate records. Before the voting and merging steps, the same videos gave a far-away partial reading (`01KN`) and several broken tracks; these were merged into the correct records. The only plate below 100% confidence was the difficult `KR803LG` (93%). All 45 records of this run, with their photos, are included as samples in `docs/samples` (`records.csv`).

#figure(
  image("figures/plates_read.jpg", width: 100%),
  caption: [Successful readings: the 15 saved plate photos with the recognised text.],
) <fig:plates>

*Vehicles without a readable plate.* The other 30 records are vehicles recorded as unrecognized: 22 passed sideways with no plate in view, and 8 had a plate that was detected but never readable (@fig:unreadable; a person cannot read them either). All 30 photos show real vehicles, so no record was made for something that is not a vehicle. Two of them are duplicates (@fig:failures): a black car on `src7` that stood near the edge while people walked past was recorded twice, and a white van on `src20` was most likely recorded twice, once as a truck and once as a car, because its type and colour changed between the two photos. The duplicate rate among the plate-less records is therefore about 2 in 30.

#figure(
  image("figures/plates_unreadable.jpg", width: 100%),
  caption: [Plates detected but not readable (motion blur, distance); recorded as unrecognized.],
) <fig:unreadable>

#figure(
  image("figures/failures.jpg", width: 100%),
  caption: [Failure cases: duplicate records and wrong type or colour.],
) <fig:failures>

*Vehicle details.* Colour was right for most vehicles but is fooled by objects in the crop (the yellow barrier post made a white van "yellow"). No vehicle was renamed "van", although two white vans appear in the videos: the zero-shot van check is too cautious at its 70% threshold. Makes were given only when the model was at least 50% sure; most given makes were right, but some were wrong (for example a Mazda named Opel), so make should be treated as a hint.

*Speed.* @tab:speed shows the measured time per step. The 14 videos took 819 s in total (8.7 frames per second, 5.5--12.2 depending on the video): a 20-second video takes 40--90 s. Speed varied with the number of vehicles in view and with other load on the machine; re-running two of the slower videos alone gave 7--8 fps. Live cameras run at about 12 fps and simply skip frames to stay current.

#figure(
  table(
    columns: (1fr, auto),
    align: (left, right),
    stroke: 0.5pt,
    inset: 4pt,
    [*Step*], [*Time*],
    [Plate detection], [≈ 30 ms / frame],
    [OCR (per plate read)], [≈ 3--7 ms],
    [Vehicle detection], [≈ 80--130 ms, every 3rd frame],
    [Annotated video (own thread)], [≈ 65 ms / frame],
    [Colour, make (once per record)], [≈ 65 ms],
  ),
  caption: [Measured processing time.],
) <tab:speed>

*Dashboard and database.* The full workflow was tested in a browser (Chromium, light and dark theme): uploading a video with progress, a corrupted file (rejected with a clear message), live view while processing, the blacklist alert, search and filters, manual review, CSV export of the filtered records, the annotated video with seeking, and live cameras (an MJPEG IP-camera stream, and an unreachable address that kept retrying). Records, jobs and lists were kept across restarts, and a database from the previous version was upgraded in place.

#figure(
  image("figures/dashboard_overview.jpg", width: 100%),
  caption: [Dashboard: overview and live view.],
) <fig:dash>

= Limitations and Future Work

- *Test data.* 14 short daytime videos from one site, Polish plates, mostly one vehicle at a time. Night, rain, heavy traffic and other countries' plates were not tested; the numbers above should not be read as general accuracy.
- *Vehicles without a plate.* Vehicles passing sideways are recorded as unrecognized. Measuring vehicle detection fully would need every vehicle annotated in every video.
- *Speed.* About 9 frames per second on a CPU for uploaded files (a 25 fps video takes 2--4.5 times its length). A GPU, or running vehicle detection less often, would be faster.
- *Plate-less vehicles.* A vehicle that stands still at the frame edge, or whose type and colour change between photos, can be recorded twice; vans are not yet recognised as vans.
- *Make and colour* are zero-shot guesses: colour is fooled by leaves or people in front of the car, and make is left empty when unsure. The MobileCLIP licence allows research use only.
- *Security.* No user accounts; keep the server on the local machine or behind a protected network.

= Conclusion

The system meets the complete workflow of the assignment: upload a traffic video, detect and track vehicles, detect and read plates, store records and photos, search and filter them in the dashboard, and export CSV. It adds live cameras and a whitelist/blacklist from the second phase. Reading each plate many times and voting, marking uncertain results for review, and merging duplicates make the records reliable on the test videos, and everything runs locally on a CPU with a small set of dependencies.
