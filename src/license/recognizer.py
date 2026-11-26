"""Find plates in a frame and read their text with two small neural networks.

The detector (YOLO) returns plate boxes; the OCR model reads one plate crop.
Both are ONNX models that run locally on the CPU."""

import numpy as np
import onnxruntime as ort
from fast_alpr import ALPR

from . import config


class Recognizer:
    def __init__(self):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = config.THREADS
        opts.inter_op_num_threads = 1
        self._alpr = ALPR(
            detector_sess_options=opts,
            ocr_sess_options=opts,
            detector_model=config.DETECTOR_MODEL,
            ocr_model=config.OCR_MODEL,
            detector_conf_thresh=config.DETECT_CONF,
        )

    def detect(self, frame):
        """Plate boxes in the frame as (x1, y1, x2, y2) integer tuples."""
        h, w = frame.shape[:2]
        boxes = []
        for d in self._alpr.detector.predict(frame):
            b = d.bounding_box
            x1, y1 = max(0, int(b.x1)), max(0, int(b.y1))
            x2, y2 = min(w, int(b.x2)), min(h, int(b.y2))
            if x2 - x1 > 4 and y2 - y1 > 2:
                boxes.append((x1, y1, x2, y2))
        return boxes

    def read(self, frame, box):
        """Read one plate. Returns (text, confidence 0-1) or None."""
        x1, y1, x2, y2 = box
        # A little margin so edge characters are not cut off
        mx, my = int(0.04 * (x2 - x1)), int(0.10 * (y2 - y1))
        h, w = frame.shape[:2]
        crop = frame[max(0, y1 - my):min(h, y2 + my), max(0, x1 - mx):min(w, x2 + mx)]
        if crop.size == 0:
            return None
        res = self._alpr.ocr.predict(crop)
        if res is None or not res.text:
            return None
        # Keep real characters only (the model pads short plates with "_")
        probs = res.confidence if isinstance(res.confidence, list) else [res.confidence] * len(res.text)
        kept = [(ch, p) for ch, p in zip(res.text.upper(), probs) if ch.isalnum()]
        if not kept:
            return None
        text = "".join(ch for ch, _ in kept)
        return text, float(np.mean([p for _, p in kept]))
