"""Vehicles: find them in every frame, and describe each recorded one.

- VehicleDetector: an object detector trained on COCO (RF-DETR nano) finds
  cars, motorcycles, buses and trucks. It runs on every VEHICLE_EVERY-th
  frame and feeds the ByteTrack tracker.
- Describer: runs once per record, in the background, on the vehicle photo.
  A CLIP model compares the photo with short text descriptions ("a photo of
  a red car", "a photo of a Toyota car", ...) and picks the closest. This
  gives the colour and the make, and tells vans from cars (COCO has no van
  class). No training is needed; the lists live in config.py.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
from open_image_models import create_detector
from tokenizers import Tokenizer

from . import config

MAKE_PROMPTS = ("a photo of a {make}.", "a photo of a {make} {kind}.", "a {make} {kind} on the road.")
COLOUR_PROMPT = "a photo of a {colour} {kind}."
VAN_PROMPTS = {"other": ("a photo of a car.", "a photo of a passenger car.",
                         "a photo of a truck.", "a photo of a lorry."),
               "van": ("a photo of a van.", "a photo of a delivery van.", "a photo of a minivan.")}


@dataclass
class Vehicle:
    box: tuple | None = None      # x1, y1, x2, y2 in the frame
    type: str | None = None
    colour: str | None = None
    make: str | None = None
    make_conf: float = 0.0


def session_options(threads=config.THREADS):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return opts


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _download(repo, name):
    """Path of a model file, downloaded only the first time."""
    try:
        return hf_hub_download(repo, name, local_files_only=True)
    except LocalEntryNotFoundError:
        return hf_hub_download(repo, name)


class VehicleDetector:
    def __init__(self):
        # A low threshold on purpose: ByteTrack uses weak detections too
        self._model = create_detector(config.VEHICLE_MODEL, conf_thresh=config.TRACK_LOW,
                                      sess_options=session_options(config.VEHICLE_THREADS))

    def detect(self, frame):
        """List of (box, confidence, label) for the vehicle classes we count."""
        h, w = frame.shape[:2]
        out = []
        for d in self._model.predict(frame):
            if d.label in config.VEHICLE_TYPES:
                b = d.bounding_box
                box = (max(0, int(b.x1)), max(0, int(b.y1)), min(w, int(b.x2)), min(h, int(b.y2)))
                if box[2] - box[0] > 8 and box[3] - box[1] > 8:
                    out.append((box, float(d.confidence), d.label))
        return out

    def find(self, frame, plate_box):
        """Box and type of the vehicle that carries the plate, or (None, None).
        Used when the tracker had no vehicle for a plate."""
        cx, cy = (plate_box[0] + plate_box[2]) / 2, (plate_box[1] + plate_box[3]) / 2
        best = None
        for box, conf, label in self.detect(frame):
            if conf >= config.TRACK_HIGH and box[0] <= cx <= box[2] and box[1] <= cy <= box[3]:
                area = (box[2] - box[0]) * (box[3] - box[1])
                if best is None or area < best[0]:       # innermost vehicle wins
                    best = (area, box, label)
        return (best[1], best[2]) if best else (None, None)


class Clip:
    """Zero-shot image classifier: picks the text label closest to an image."""

    def __init__(self):
        paths = [_download(config.CLIP_REPO, f) for f in config.CLIP_FILES]
        opts = session_options()
        self._vision = ort.InferenceSession(paths[0], opts, providers=["CPUExecutionProvider"])
        self._text = ort.InferenceSession(paths[1], opts, providers=["CPUExecutionProvider"])
        self._tok = Tokenizer.from_file(paths[2])

    def embed_text(self, prompts):
        ids = np.zeros((len(prompts), 77), np.int64)
        for i, p in enumerate(prompts):
            t = self._tok.encode(p).ids[:77]
            ids[i, :len(t)] = t
        return _unit(self._text.run(None, {"input_ids": ids})[0])

    def embed_image(self, bgr):
        # Resize the short side, centre-crop, RGB in 0..1 (what the model expects)
        s = config.CLIP_SIZE
        h, w = bgr.shape[:2]
        k = s / min(h, w)
        img = cv2.resize(bgr, (max(s, round(w * k)), max(s, round(h * k))),
                         interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
        y, x = (img.shape[0] - s) // 2, (img.shape[1] - s) // 2
        img = cv2.cvtColor(img[y:y + s, x:x + s], cv2.COLOR_BGR2RGB)
        pixels = (img.astype(np.float32) / 255).transpose(2, 0, 1)[None]
        return _unit(self._vision.run(None, {"pixel_values": pixels})[0][0])

    def labels(self, names, prompts, kind=""):
        """Text embedding per label, averaged over several phrasings."""
        return np.stack([_unit(self.embed_text([p.format(make=n, colour=n, kind=kind)
                                                for p in prompts]).mean(0))
                         for n in names])


def _pick(names, label_emb, image_emb):
    p = np.exp(100 * (label_emb @ image_emb))       # CLIP's usual scale
    p /= p.sum()
    i = int(p.argmax())
    return names[i], float(p[i])


class Describer:
    def __init__(self):
        self._clip = Clip()
        makes = {"car": config.MAKES, "van": config.MAKES, "motorcycle": config.MOTORCYCLE_MAKES,
                 "bus": config.BUS_TRUCK_MAKES, "truck": config.BUS_TRUCK_MAKES}
        # The text side is fixed, so it is computed once here
        self._makes = {k: (names, self._clip.labels(names, MAKE_PROMPTS, k))
                       for k, names in makes.items()}
        self._colours = {k: self._clip.labels(config.COLOURS, [COLOUR_PROMPT], k) for k in makes}
        self._van = (list(VAN_PROMPTS),
                     np.stack([_unit(self._clip.embed_text(p).mean(0)) for p in VAN_PROMPTS.values()]))

    def describe(self, frame, box, kind):
        """Type, colour and make of the vehicle in `box`."""
        x1, y1, x2, y2 = (int(v) for v in box)
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            return Vehicle(box, kind)
        emb = self._clip.embed_image(crop)
        if kind in ("car", "truck"):
            name, p = _pick(*self._van, emb)
            if name == "van" and p >= config.VAN_MIN_CONF:
                kind = "van"
        colour, _ = _pick(config.COLOURS, self._colours[kind], emb)
        make, conf = _pick(*self._makes[kind], emb)
        return Vehicle(tuple(box), kind, colour, make if conf >= config.MAKE_MIN_CONF else None, conf)
