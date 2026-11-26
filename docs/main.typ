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
    This project detects moving vehicles, locates the license plate of each vehicle, and uses optical character recognition (OCR) to read the text on the plate.

    The recognized plate text is tracked across video frames and stored together with the time stamp of detection and relevant features of the plate, such as the recognition confidence and the number of readings that support it.
  ],
  authors: (
    (
      name: "Spandan Guragain",
      email: "spandan@3o14.com"
    ),
  ),
  index-terms: ("OCR", "OpenCV", "Real Time"),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

#outline()

= Introduction

Automatic license plate recognition (ALPR) is a core task in camera-based security, access control and traffic monitoring. A system has to find the plate in a cluttered scene, read its characters reliably despite blur, glare and small image size, and record each vehicle once, with a time stamp.

This project utilizes the following components to successfully determine and log the license plates of the moving vehicles:

+ OpenCV for image processing, contour analysis and video input.
+ PyTesseract, a Python wrapper for the Tesseract OCR engine @smith2007, for reading the plate characters.
+ NumPy for array operations.
+ imutils for portable contour handling.

The program accepts either video files or a stream URL, processes the frames one at a time, and appends one row per recognized vehicle to a CSV log.

= System Overview

@fig:pipeline lists the stages of the pipeline. The first stages are classical image-processing steps that propose candidate regions. Later stages verify a candidate by finding character-like shapes in it, read the text, and combine the readings of several frames into a single result.

#figure(
  table(
    columns: (auto, 1fr),
    align: (center, left),
    stroke: 0.5pt,
    [*\#*], [*Stage*],
    [1], [Gaussian blur and grayscale conversion],
    [2], [Vertical edge detection (Sobel)],
    [3], [Otsu binarization],
    [4], [Morphological closing],
    [5], [Contour extraction],
    [6], [Minimum-area rectangle validation],
    [7], [Character localization and tight crop],
    [8], [OCR with Tesseract (several variants)],
    [9], [Tracking and per-vehicle voting],
    [10], [Time-stamped CSV logging],
  ),
  caption: [Processing stages of the recognition pipeline.],
) <fig:pipeline>

= Methodology

== Preprocessing

Each frame is smoothed with a Gaussian filter to suppress sensor noise and converted to grayscale. Smoothing first reduces the number of spurious edges found in the next step.

== Vertical Edge Detection

Plate characters are tall, thin strokes, so a plate region contains many vertical edges. The horizontal derivative is computed with a $3 times 3$ Sobel kernel:

$ G_x = mat(-1, 0, 1; -2, 0, 2; -1, 0, 1) ast.op I $

The absolute value of $G_x$ is taken and scaled to 8 bits, because the signed response would otherwise lose the dark-to-light edges.

== Binarization

The edge image is binarized with Otsu's method @otsu1979, which selects the threshold $t$ that maximizes the between-class variance:

$ sigma_b^2 (t) = omega_0 (t) omega_1 (t) [mu_0 (t) - mu_1 (t)]^2 $

where $omega_i$ and $mu_i$ are the probability and mean intensity of the two classes. No threshold has to be chosen by hand, so the method adapts to different lighting.

== Morphological Closing

A closing operation (dilation followed by erosion) with a wide, short rectangular kernel joins the vertical edges of neighbouring characters into one solid block, which reveals the rectangular body of the plate. The kernel width is derived from the frame width so that the same code works at different resolutions.

== Contours and Candidate Validation

Contours of the closed image are extracted with the border-following algorithm of Suzuki and Abe @suzuki1985. The minimum-area rotated rectangle of each contour, with width $w$ and height $h$ ($w >= h$), is kept only if

$ A_"min" <= w h <= A_"max", quad r_"min" <= w / h <= r_"max" $

and its tilt is below $15 degree$. The area limits are expressed as fractions of the frame width, so they scale with the video resolution. Candidates are then rotated to be level and cropped with padding, so that no character is cut off.

== Character Localization

Edge-based candidates also include grilles, lane markings, foliage and brick joints. To reject them, each candidate is binarized in several ways (Otsu and adaptive, both polarities), and connected components are filtered. A set of components is accepted as a plate only if it contains enough characters, with similar heights, a common baseline and even spacing. Rows are grouped by vertical position, which also supports two-line plates. The accepted characters define a tight crop; the frame, screws and background are discarded before OCR, and mostly green candidates are rejected as vegetation.

== Optical Character Recognition

The tight crop is resized so that the character height suits Tesseract, and three variants are prepared: contrast-limited adaptive histogram equalization (CLAHE) @pizer1987, an Otsu version, and an adaptive-threshold version. Tesseract runs on each variant with a whitelist of letters and digits, using single-line or block mode depending on the number of rows. No country-specific format is assumed.

== Tracking and Voting

A single frame of a moving vehicle is often blurred, so one reading is not trusted. A nearest-centre tracker assigns an identifier to each plate across frames. For each track, the readings $t_k$ with confidence $w_k$ are combined by a weighted per-position vote:

$ hat(c)_i = op("arg max", limits: #true)_c sum_k w_k bb(1) [t_(k,i) = c] $

An agreement score measures how consistent the readings are with the voted text of length $L$ over $N$ readings:

$ a = 1 / N sum_(k=1)^N 1 / L sum_(i=1)^L bb(1) [t_(k,i) = hat(c)_i] $

A track is logged only if it persisted for several detections, has enough readings above a confidence floor, and reaches a minimum agreement. Random shapes such as foliage give different text in each frame, so they fail this test.

== Logging

When a track ends, its time stamp (the time of first appearance), source, plate text, mean confidence and number of readings are appended to a CSV file. A cooldown prevents the same plate from being logged twice for one pass.

= Implementation

The program is written in Python 3.12 and managed with `uv`. Videos placed in the `testset` directory are processed by default; a file path or a stream URL can be given instead. For video files, time stamps are the start time of the run plus the position in the video; for streams, the wall-clock time is used.

@tab:params summarizes the main parameters.

#figure(
  table(
    columns: (1fr, auto),
    align: (left, left),
    stroke: 0.5pt,
    [*Parameter*], [*Value*],
    [Plate width / frame width], [0.04--0.13],
    [Aspect ratio range], [2.0--6.0],
    [Maximum tilt], [$15 degree$],
    [Region of interest (vertical)], [20%--80%],
    [Minimum characters], [5],
    [Detection interval], [every 2 frames],
    [Minimum detections per track], [4],
    [Minimum readings per track], [3],
    [Minimum agreement], [0.6],
    [Minimum OCR confidence], [40],
    [Lost-track timeout], [20 frames],
    [Re-log cooldown], [30 s],
  ),
  caption: [Main parameters of the pipeline.],
) <tab:params>

The candidate validation is the central filter of the detection stage. A simplified version is shown in @lst:validate.

#figure(
  ```python
  def validate(rect):
      _, (w, h), angle = normalize(rect)
      if w == 0 or h == 0:
          return False
      return (
          MIN_AREA <= w * h <= MAX_AREA
          and MIN_RATIO <= w / h <= MAX_RATIO
          and abs(angle) <= MAX_TILT
      )
  ```,
  caption: [Candidate rectangle validation.],
) <lst:validate>

= Preliminary Observations

Development on sample footage showed the following behaviours of the classical approach:

- *Plate size depends on the scene.* Plates were about 2--5% of the frame width in wide traffic footage and about 7% in a close-up barrier scene, so fixed pixel limits fail. The limits were therefore made relative to the frame width.
- *Blur confuses similar characters.* On low-resolution crops, an `A` can be read as `4` and a `5` as `9`. Removing the dark plate frame before thresholding and voting across frames reduced such errors.
- *Cropping decides OCR quality.* Characters at the plate ends are the blurriest and are the first to be dropped by strict filters, which leaves them out of the OCR input entirely.
- *Busy scenes cause false detections.* Foliage and paving produce vertical edges that resemble plates, which motivated the character-shape checks, the colour check and the agreement filter.

= Evaluation Plan

#text(fill: red)[TODO: no quantitative results have been measured yet.]

The system will be evaluated on the videos in the test set with manually labelled ground truth. The planned metrics are:

+ Detection recall and precision per vehicle.
+ Character-level accuracy of the logged text.
+ Exact-match accuracy of the whole plate.
+ Processing speed in frames per second on a CPU.

= Limitations and Future Work

The pipeline depends on hand-tuned thresholds, and Tesseract's general-purpose English model is not trained on plate fonts, so accuracy drops on small, blurred or unusual plates. The centroid tracker can break when vehicles move quickly or cross each other.

Future work will compare this baseline with a trained plate detector and a plate-specific OCR model, replace the tracker with a standard multi-object tracker, and add support for reconnecting to dropped streams.

= Conclusion

This report described a real-time license plate recognition pipeline built from OpenCV and Tesseract. Candidate regions are found with edge, threshold and morphology operations, verified by character-shape analysis, read with several OCR variants, and merged over time by per-vehicle voting before being logged with a time stamp. The system provides a working baseline, and the planned evaluation will quantify its accuracy and show where a learned detector or OCR model should replace the hand-built stages.
