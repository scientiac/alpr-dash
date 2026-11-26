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

#show: ieee.with(
  title: [Real Time License Plate Recognition System],
  abstract: [
    This project watches a video or a live camera stream, finds the license plate of every vehicle, reads the text on it and logs it with a time stamp.

    Plates are found and read by two small neural networks that run locally on an ordinary CPU in real time. Each plate is followed across frames and read many times; the readings are combined by a vote, so a single blurred frame cannot cause a wrong result. A simple web dashboard shows the live video, the plates in view and the logged vehicles, and exports the log as a CSV file.
  ],
  authors: (
    (
      name: "Spandan Guragain",
      email: "spandan@3o14.com"
    ),
  ),
  index-terms: ("License plate recognition", "OCR", "YOLO", "Real Time"),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

#outline()

= Introduction

Automatic license plate recognition (ALPR) is used for parking, access control and traffic monitoring. A system has to find a small plate in a busy picture, read its characters despite blur and glare, and record each vehicle once, with a time stamp.

The first prototype of this project used classical image processing (edge detection, thresholds and contour shapes) with the general-purpose Tesseract OCR engine @smith2007. It needed many hand-tuned numbers, confused similar characters such as `A`/`4` and `5`/`9`, and mistook grilles and leaves for plates. The current version replaces those hand-built steps with two models trained specifically on license plates, which is both simpler and far more accurate.

The project uses:

+ *fast-alpr*, which bundles a YOLOv9 plate detector @wang2024yolov9 and a plate OCR model based on a compact transformer @hassani2021cct. Both run with ONNX Runtime on the CPU.
+ *OpenCV* for reading video files and streams and for drawing on frames.
+ Python's built-in web server for the dashboard; no web framework is needed.

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
    [4], [Read the text of each plate (OCR)],
    [5], [Combine the readings of each plate by voting],
    [6], [When a plate leaves the view, log it to CSV],
    [7], [Draw boxes and send the frame to the dashboard],
  ),
  caption: [Processing steps for each frame.],
) <fig:pipeline>

== Finding plates

The detector looks at the whole frame once and returns a box around each plate it finds, with a confidence score. It was trained on many thousands of plate photos, so it does not need rules about plate size, colour or shape. Boxes below 40% confidence are ignored.

== Following each vehicle

A moving car is seen in many frames. The tracker links a new box to an existing plate when the two boxes overlap or their centres are close (within about one and a half plate widths). Otherwise a new plate track starts. A track ends when its plate has not been seen for 15 frames.

== Reading the text

Each plate box is cut out with a small margin and given to the OCR model, which returns the characters and how sure it is about each one. Readings that are too short, too long, or less than 50% certain are thrown away. To save time, a plate is read only every second frame, and less often once its readings all agree.

== Voting

A single frame can be blurred, or the car can still be far away so that a character is missed. The system therefore keeps the 12 most confident readings of each plate and votes:

+ The most common text length wins.
+ For each character position, the character with the highest total confidence wins.

For example, the readings `KR154K`, `KR154GK`, `KR154GK` give `KR154GK`. The *agreement* is the share of characters, over all readings, that match the voted text. A plate is logged only if it was detected at least 3 times, read at least twice, and its agreement is at least 60%. Random shapes give different text each time, so they fail this check.

== Logging

When a track ends, one row is appended to `plates.csv`: time of first appearance, source, plate text, confidence and number of readings. If the same plate is logged again from the same source within 30 seconds (for example because the car was hidden for a moment), the duplicate is skipped. For video files the time is the start of the run plus the position in the video; for streams it is the current clock time.

== Staying real time

Detection is the slowest step (about 30 ms per frame). When a video file starts to fall behind its natural speed, detection is skipped on the next frame and the boxes stay where they were; this keeps the playback smooth. For live streams a background thread always keeps only the newest frame, so the system never lags behind the camera. Each model uses 4 CPU threads, which is as fast as using all cores but uses far less CPU.

= Dashboard

Running the program opens a dashboard at `http://127.0.0.1:8000`. It shows:

- the live video with a box and the current reading over each plate (green once the plate is confirmed);
- the number of logged vehicles, unique plates and plates in view;
- the plates currently in view and the table of logged vehicles;
- an *Export CSV* button that downloads the logged table from the browser.

= Usage

```
uv run license                  # all videos in src/testset
uv run license video.mp4        # one file
uv run license rtsp://camera/1  # a live stream
uv run license --fast           # files as fast as possible
uv run license --no-web         # console and CSV only
```

The models (about 13 MB) are downloaded on the first run and cached; after that everything runs offline. All settings are in `config.py`.

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
    [`tracker.py`], [Tracking, voting, agreement],
    [`store.py`], [CSV log and list for the dashboard],
    [`pipeline.py`], [The per-frame loop and drawing],
    [`web/`], [Dashboard server and page],
    [`__init__.py`], [Command line entry point],
  ),
  caption: [Source files in `src/license`.],
) <tab:files>

= Results

The system was run on the five test videos (1280×720, 25 fps) on a 16-thread CPU without a GPU.

#figure(
  table(
    columns: (1fr, auto, auto),
    align: (left, left, right),
    stroke: 0.5pt,
    [*Video*], [*Logged plate*], [*Conf.*],
    [`agh_src5`], [`KR439HA`], [100%],
    [`agh_src6`], [`KR537CW`], [100%],
    [`agh_src8`], [`KR642LU`], [100%],
    [`agh_src10`], [`KR154GK`], [100%],
    [`agh_src13`], [`KR992ES`], [100%],
  ),
  caption: [Logged plates on the test set.],
) <tab:results>

Every vehicle was logged exactly once with the correct text and no false entries.

#figure(
  table(
    columns: (1fr, auto),
    align: (left, right),
    stroke: 0.5pt,
    [*Step*], [*Time per frame*],
    [Read frame], [≈ 2 ms],
    [Detect plates], [≈ 30 ms],
    [OCR (all plates)], [≈ 7 ms],
    [Draw and encode], [≈ 3 ms],
  ),
  caption: [Measured processing time.],
) <tab:speed>

Without the dashboard and without speed limiting, the whole test set is processed at about 27 frames per second. With the dashboard, the videos play at their full 25 frames per second.

= Limitations and Future Work

The test set is small: five videos, each with one main vehicle, all with Polish plates. Larger and more varied footage (night, rain, several cars at once, other countries) is needed to measure accuracy properly. Very distant plates in the background are detected but too small to read, so they are not logged. The tracker uses only box position; fast vehicles crossing each other could swap tracks. A GPU, or a smaller detector input, would allow several cameras on one machine.

= Conclusion

Replacing the hand-tuned image-processing pipeline with a trained plate detector and a plate-specific OCR model made the system simpler and much more accurate, while still running in real time on a CPU. Tracking and voting over many frames remove the errors of single blurred frames, and the web dashboard makes the results easy to watch and export.
