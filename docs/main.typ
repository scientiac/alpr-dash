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
  title: [Real Time License Plate Recognition System],
  abstract: [
    This project watches a video or a live camera stream, finds the license plate of every vehicle, reads the text on it and records it with a time stamp, the direction of travel, the vehicle's type, colour and make, and photos of the plate and the vehicle.

    Plates are found and read by two small neural networks that run locally on an ordinary CPU in real time. Each plate is followed across frames and read many times; the readings are combined by a vote, so a single blurred frame cannot cause a wrong result. Two more models describe each vehicle once it has passed. Everything is stored in a SQLite database and a folder of photos, and a simple web dashboard shows the live video and the records.
  ],
  authors: (
    (
      name: "Spandan Guragain",
      email: "spandan@3o14.com"
    ),
  ),
  index-terms: ("License plate recognition", "OCR", "YOLO", "CLIP", "SQLite", "Real Time"),
  figure-supplement: [Fig.],
)

#outline()

= Introduction

Automatic license plate recognition (ALPR) is used for parking, access control and traffic monitoring. A system has to find a small plate in a busy picture, read its characters despite blur and glare, and record each vehicle once, with a time stamp.

The first prototype of this project used classical image processing (edge detection, thresholds and contour shapes) with the general-purpose Tesseract OCR engine. It needed many hand-tuned numbers, confused similar characters such as `A`/`4` and `5`/`9`, and mistook grilles and leaves for plates. The current version replaces those hand-built steps with models trained for the job, which is both simpler and far more accurate.

For every vehicle the system now records:

- the plate text, with a confidence and the number of readings;
- the time, the source video or camera, and whether the vehicle was coming in or going out;
- the vehicle type, colour and make;
- a photo of the plate (to check the reading by eye) and a photo of the whole vehicle.

= Models

Four neural networks are used (@tab:models). All are ONNX files that run with ONNX Runtime on the CPU. They are downloaded automatically on the first run and cached in the home folder; after that the system works offline.

#figure(
  table(
    columns: (auto, 1fr, auto),
    align: (left, left, right),
    stroke: 0.5pt,
    [*Job*], [*Model*], [*Size*],
    [Find plates], [YOLOv9-t, 512 px, trained on plates (`fast-alpr`)], [8 MB],
    [Read plates], [CCT-S v2 global plate OCR (`fast-alpr`)], [5 MB],
    [Find vehicle, type], [RF-DETR nano, 384 px, trained on COCO (`open-image-models`)], [115 MB],
    [Colour, make], [MobileCLIP-B (LT), 8-bit (Apple, via Hugging Face `Xenova/mobileclip_blt`)], [150 MB],
  ),
  caption: [Models used.],
) <tab:models>

The first two run on every frame. The last two run only once per recorded vehicle, in a background thread, so they do not slow down the live video.

= How It Works

Every frame passes through the steps in @fig:pipeline.

#figure(
  table(
    columns: (auto, 1fr),
    align: (center, left),
    stroke: 0.5pt,
    [*\#*], [*Step*],
    [1], [Read a frame from the file or stream],
    [2], [Detect plates (YOLO) #sym.arrow boxes],
    [3], [Match boxes to plates seen in earlier frames],
    [4], [Read the text of each plate (OCR); keep the frame of the clearest reading],
    [5], [Combine the readings of each plate by voting],
    [6], [When a plate leaves the view, pass it to the recorder],
    [7], [Draw boxes and send the frame to the dashboard],
  ),
  caption: [Processing steps for each frame.],
) <fig:pipeline>

The recorder (@sec:recorder) then describes the vehicle, removes duplicates and saves the record.

== Finding plates

The detector looks at the whole frame once and returns a box around each plate it finds, with a confidence score. It was trained on many thousands of plate photos, so it does not need rules about plate size, colour or shape. Boxes below 40% confidence are ignored.

== Following each vehicle

A moving car is seen in many frames. The tracker links a new box to an existing plate when the two boxes overlap or their centres are close (within about one and a half plate widths). Otherwise a new plate track starts. A track ends when its plate has not been seen for 15 frames. The track remembers where the plate was in every frame; this is used for the direction and for finding duplicates.

== Reading the text

Each plate box is cut out with a small margin and given to the OCR model, which returns the characters and how sure it is about each one. Readings that are too short, too long, or less than 50% certain are thrown away. To save time, a plate is read only every second frame. Whenever a reading is the clearest so far (most certain, then largest plate), a copy of that frame is kept for the photos.

== Voting

A single frame can be blurred, or the car can still be far away so that a character is missed. The system therefore keeps the 12 most confident readings of each plate and votes:

+ The most common text length wins.
+ For each character position, the character with the highest total confidence wins.

For example, the readings `KR154K`, `KR154GK`, `KR154GK` give `KR154GK`. The *agreement* is the share of characters, over all readings, that match the voted text. A plate is recorded only if it was detected at least 3 times, read at least twice, and its agreement is at least 60%. Random shapes give different text each time, so they fail this check.

== Direction (incoming or outgoing)

The direction comes from how the plate moved while it was tracked. The first and last few boxes of the track are averaged and compared:

- the plate grew by at least 15%, or moved down the picture by at least 5% of its height: the vehicle came *towards* the camera;
- the plate shrank by the same amount, or moved up: it went *away* from the camera;
- otherwise (for example, a car crossing the picture) the direction is left empty.

By default, towards the camera means *in*. If the camera faces the exit, setting `TOWARDS_CAMERA = "out"` in `config.py` swaps them.

== Staying real time

Plate detection is the slowest step on each frame (about 30 ms). When a video file starts to fall behind its natural speed, detection is skipped on the next frame and the boxes stay where they were; this keeps the playback smooth. For live streams a background thread always keeps only the newest frame, so the system never lags behind the camera. Each model uses 4 CPU threads, which is as fast as using all cores but uses far less CPU.

= Recorder <sec:recorder>

Finished plates are put in a queue. A background thread takes them one at a time and does the slower work, about 0.2 s per vehicle.

== Vehicle type

The RF-DETR detector is run on the kept frame. It finds every car, motorcycle, bus and truck. The vehicle is the one whose box contains the centre of the plate; if several do (a car in front of a bus), the smallest box wins. Its class is the vehicle type and its box is used for the vehicle photo. If no vehicle is found, the whole frame is saved and type, colour and make stay empty.

== Colour and make

The vehicle photo is given to MobileCLIP, a model that turns both pictures and short sentences into lists of numbers that are close together when they mean the same thing. The photo is compared with sentences such as "a photo of a red car" or "a photo of a Toyota car", and the closest sentence wins. This is called zero-shot classification: no training is needed, and the lists of colours and makes are plain text in `config.py`, so makes common in another country can simply be added. Each make is described in three different sentences and their average is used, which makes the choice steadier. Separate make lists are used for cars, motorcycles and buses or trucks.

Colour is usually right, but leaves or shadows in front of the car can change it. Make is harder: the test videos show old European cars in low resolution, and on single photos the model named the right make for only about half of them. The make is therefore saved only when the model is at least 50% sure; otherwise it is left empty. With this rule every make it gave on the test set was right (see Results).

== Duplicates

The same vehicle can be recorded twice: its track may break when it is hidden for a moment, a character may be misread in one of the two parts, or a car may carry a second plate-like sign. Two records are treated as the same vehicle when they come from the same source at about the same time (within 30 seconds) and:

- the plate text is the same, or
- the vehicle looks the same (same type and colour) and either the plate text differs in only one character, or one plate was inside the other's vehicle box in the same frame.

Only the record with the better reading (higher confidence, then more readings) is kept; the other is dropped, or replaces the first one in the database if it is better.

= Storage

Everything is stored in the `data` folder, which is created on the first run.

== Database

The records are kept in a SQLite database, `data/plates.db`. SQLite is part of Python, needs no server, and the whole database is one file that can be copied or opened with any SQLite tool. Each record is one row of the `sightings` table (@tab:schema).

#figure(
  table(
    columns: (auto, 1fr),
    align: (left, left),
    stroke: 0.5pt,
    [*Column*], [*Meaning*],
    [`id`], [Record number],
    [`time`], [When the plate was first seen],
    [`source`], [Video file or stream],
    [`plate`], [Plate text],
    [`confidence`], [Average OCR confidence, 0--100],
    [`readings`], [Readings that were combined],
    [`direction`], [`in`, `out` or empty],
    [`vehicle_type`], [`car`, `motorcycle`, `bus`, `truck` or empty],
    [`colour`], [Vehicle colour],
    [`make`], [Vehicle make, empty if unsure],
    [`plate_image`], [Path of the plate photo],
    [`vehicle_image`], [Path of the vehicle photo],
  ),
  caption: [Columns of the `sightings` table.],
) <tab:schema>

The database uses write-ahead logging, so the dashboard can read while the recorder writes.

== Photos

Photos are not put inside the database. They are saved as JPEG files in `data/images`, one folder per day, and the database stores their path inside that folder:

```
data/images/2026-11-27/
  114700_KR992ES_plate_57aa44.jpg
  114700_KR992ES_vehicle_45c574.jpg
```

Plain files keep the database small and fast, are easy to look at, back up or delete, and can be sent straight to the browser. The plate photo is the plate with a small margin; the vehicle photo is the vehicle's box, both from the frame with the clearest reading.

== Using the data

The database can be read with any SQLite tool, for example:

```
sqlite3 data/plates.db \
  "SELECT time, plate, direction, colour, make
   FROM sightings ORDER BY time DESC LIMIT 10"
```

This replaces the CSV file of the earlier version: the data is kept in one place, can be searched, and stays consistent when a duplicate is replaced.

= Dashboard

Running the program opens a dashboard at `http://127.0.0.1:8000`. It shows:

- the live video with a box and the current reading over each plate (green once the plate is confirmed);
- the number of recorded vehicles, how many came in and went out, and the plates in view;
- the table of recorded vehicles from the database, newest first, with a small plate photo, the vehicle description and the direction;
- clicking a row opens the vehicle photo, the plate photo and all details.

= Installation

== System requirements

- A 64-bit Linux, macOS or Windows computer. No GPU is needed; the results below are from a 16-thread CPU.
- About 1 GB of free disk space for the Python packages and the models, plus space for the photos (about 30 kB per vehicle).
- An internet connection for the first run only, to download the packages and models.
- `uv`, the Python package manager. It also installs the right Python version (3.12), so Python does not have to be installed separately.

No other system packages are needed: the OpenCV package already contains FFmpeg for reading video files and network streams, and SQLite is part of Python.

Optional tools:

- `sqlite3` (command line) to look at the database: `sudo apt install sqlite3` on Debian/Ubuntu, `brew install sqlite` on macOS.
- `typst` to rebuild this document from `docs/main.typ`.

== Steps

+ Install `uv`:
  ```
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
  On Windows:
  ```
  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
+ Get the code and enter its folder:
  ```
  git clone <repository> license
  cd license
  ```
+ Install Python 3.12 and all packages into `.venv`:
  ```
  uv sync
  ```
+ Run it once on the test videos. The first run downloads the four models (about 280 MB):
  ```
  uv run license
  ```
+ Open `http://127.0.0.1:8000` in a browser.
+ To rebuild this document: `typst compile docs/main.typ`.

== Dependencies

@tab:deps lists the Python packages the project uses directly (they are declared in `pyproject.toml`, and exact versions are pinned in `uv.lock`). The other packages `uv` installs are needed by these.

#figure(
  table(
    columns: (auto, 1fr),
    align: (left, left),
    stroke: 0.5pt,
    [*Package*], [*Used for*],
    [`fast-alpr[onnx]`], [Plate detector and OCR models; brings in `onnxruntime`, `open-image-models` (also gives the vehicle detector) and `fast-plate-ocr`],
    [`opencv-python-headless`], [Reading videos and streams, cropping, drawing, JPEG files],
    [`numpy`], [Arrays and the voting maths],
    [`tokenizers`], [Turning the colour and make sentences into model input],
    [`huggingface-hub`], [Downloading and caching the MobileCLIP model],
  ),
  caption: [Python packages.],
) <tab:deps>

The Python standard library covers the rest: `sqlite3` for the database, `http.server` for the dashboard, and `threading` and `queue` for the background work.

= Usage

#figure(
  table(
    columns: (auto, 1fr),
    align: (left, left),
    stroke: 0.5pt,
    [*Command*], [*What it does*],
    [`uv run license`], [All videos in `src/testset`],
    [`uv run license video.mp4`], [One video file],
    [`uv run license rtsp://cam/1`], [A live stream (RTSP, HTTP, ...)],
    [`--fast`], [Process files as fast as possible],
    [`--no-web`], [No dashboard; console and database only],
    [`--port 9000`], [Dashboard on another port],
  ),
  caption: [Command line.],
) <tab:usage>

Records from earlier runs stay in the database; delete the `data` folder to start again. All settings, including the lists of colours and makes and the direction, are in `config.py`.

= Code Structure

#figure(
  table(
    columns: (auto, 1fr),
    align: (left, left),
    stroke: 0.5pt,
    [*File*], [*Purpose*],
    [`config.py`], [All settings],
    [`sources.py`], [Choose inputs, lag-free stream reader],
    [`recognizer.py`], [Plate detection and OCR],
    [`tracker.py`], [Tracking, voting, agreement, direction],
    [`pipeline.py`], [The per-frame loop and drawing],
    [`vehicles.py`], [Vehicle type, colour and make],
    [`recorder.py`], [Background recording and duplicate removal],
    [`db.py`], [SQLite database and photo files],
    [`web/`], [Dashboard server and page],
    [`__init__.py`], [Command line entry point],
  ),
  caption: [Source files in `src/license`.],
) <tab:files>

= Results

The system was run on the 14 test videos (1280×720, 25 fps, 7084 frames in total) on a 16-thread CPU without a GPU. @tab:results shows the database after the run. The true make of each car was checked by eye on the saved photos.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, left, center, left, left),
    stroke: 0.5pt,
    inset: 4pt,
    [*Video*], [*Plate*], [*Dir.*], [*Colour*], [*Make (true)*],
    [`src1`], [`KMI77EV`], [out], [white], [Opel ✓],
    [`src1`], [`KR803LG`], [–], [grey], [Kia (hidden)],
    [`src4`], [`KR154GK`], [in], [red], [– (Rover)],
    [`src5`], [`KR439HA`], [in], [green ✗], [Citroen ✓],
    [`src6`], [`KR537CW`], [out], [grey], [Audi ✓],
    [`src7`], [`KR539JK`], [in], [white], [– (Nissan)],
    [`src8`], [`KR642LU`], [in], [grey], [– (Toyota)],
    [`src10`], [`KR701KN`], [in], [white], [Renault ✓],
    [`src11`], [`KR731GW`], [out], [black], [Volkswagen ✓],
    [`src12`], [`KR960MF`], [out], [black], [– (Toyota)],
    [`src13`], [`KR992ES`], [in], [blue], [– (Mazda)],
    [`src15`], [`KR2492K`], [in], [black], [– (Subaru)],
    [`src18`], [`KR3527L`], [in], [white], [Skoda ✓],
    [`src20`], [`KR6986N`], [in], [grey], [Peugeot ✓],
    [`src22`], [`KR9195N`], [in], [blue], [Opel ✓],
  ),
  caption: [Records on the test set. All plates were read at 100% confidence.],
) <tab:results>

- *Plates:* all 15 vehicles were recorded once, with the correct text, and nothing else was recorded. Without the duplicate check there would have been 21 records: broken tracks on `src15`, `src5`, `src7` and twice on `src1`, and a far-away partial reading `O1KN` of the car on `src10`, which was merged into `KR701KN`.
- *Direction:* correct for all 13 vehicles that drove towards or away from the camera (front plates "in", rear plates "out"). The car on `src1` that was only partly visible got no direction.
- *Type:* all were correctly found as cars. On `src18` the bus is detected, but its plate is never visible, so only the car is recorded.
- *Colour:* correct except `src5`, where leaves in front of a silver van made it "green". The beige Renault was called white.
- *Make:* given for 9 vehicles; all 8 that could be checked are right. For the other 6 the model was unsure and left the make empty instead of guessing.

#figure(
  table(
    columns: (1fr, auto),
    align: (left, right),
    stroke: 0.5pt,
    [*Step*], [*Time*],
    [Read frame], [≈ 2 ms / frame],
    [Detect plates], [≈ 30 ms / frame],
    [OCR (all plates)], [≈ 7 ms / frame],
    [Draw and encode], [≈ 3 ms / frame],
    [Find vehicle (RF-DETR)], [≈ 130 ms / vehicle],
    [Colour and make (MobileCLIP)], [≈ 65 ms / vehicle],
  ),
  caption: [Measured processing time.],
) <tab:speed>

Without the dashboard and without speed limiting, the whole test set (7084 frames) took 4 min 55 s including loading the models, about 24 frames per second. The vehicle models run in the background thread, so with the dashboard the videos still play at their full 25 frames per second. Both photos together take about 25 kB per vehicle.

= Limitations and Future Work

The test set is small: each video has one main vehicle, all with Polish plates, in daylight. Larger and more varied footage (night, rain, several cars at once, other countries) is needed to measure accuracy properly. Very distant plates are detected but too small to read, so they are not recorded. The tracker uses only box position; fast vehicles crossing each other could swap tracks.

The make is the weakest part. A model trained on labelled photos of the makes common where the camera is installed would do much better than the general zero-shot model, and could replace it without other changes. The direction rule assumes a camera that looks along the road; a camera that looks across it should use left/right movement instead. A GPU, or a smaller detector input, would allow several cameras on one machine.

= Conclusion

Replacing the hand-tuned image-processing pipeline with a trained plate detector and a plate-specific OCR model made the system simpler and much more accurate, while still running in real time on a CPU. Tracking and voting over many frames remove the errors of single blurred frames. Two further models, run once per vehicle in the background, add the vehicle's type, colour and make without slowing the video, and also help to recognise when the same vehicle was recorded twice. A single SQLite file and a folder of photos keep every record with the evidence needed to check it.
