"""Find the vehicle a plate belongs to and describe it: type, colour and make.

This runs once per logged plate (not on every frame), so the slower models
here do not affect the live video.

- Type: an object detector trained on COCO (RF-DETR) finds cars, motorcycles,
  buses and trucks; the vehicle is the one whose box contains the plate.
- Colour and make: a CLIP model compares the vehicle photo with short text
  descriptions ("a photo of a red car", "a photo of a Toyota car", ...) and
  picks the closest. No training is needed; the lists live in config.py.
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


@dataclass
class Vehicle:
    box: tuple | None = None      # x1, y1, x2, y2 in the frame
    type: str | None = None
    colour: str | None = None
    make: str | None = None
    make_conf: float = 0.0


def _session_options():
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = config.THREADS
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


class Clip:
    """Zero-shot image classifier: picks the text label closest to an image."""

    def __init__(self):
        paths = [_download(config.CLIP_REPO, f) for f in config.CLIP_FILES]
        opts = _session_options()
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

    def labels(self, names, prompts, kind):
        """Text embedding per label, averaged over several phrasings."""
        return np.stack([_unit(self.embed_text([p.format(make=n, colour=n, kind=kind)
                                                for p in prompts]).mean(0))
                         for n in names])


def _pick(names, label_emb, image_emb):
    p = np.exp(100 * (label_emb @ image_emb))       # CLIP's usual scale
    p /= p.sum()
    i = int(p.argmax())
    return names[i], float(p[i])


class VehicleDescriber:
    def __init__(self):
        self._detector = create_detector(config.VEHICLE_MODEL, conf_thresh=config.VEHICLE_CONF,
                                         sess_options=_session_options())
        self._clip = Clip()
        makes = {"car": config.MAKES, "motorcycle": config.MOTORCYCLE_MAKES,
                 "bus": config.BUS_TRUCK_MAKES, "truck": config.BUS_TRUCK_MAKES}
        # Text side is fixed, so it is computed once here
        self._makes = {k: (names, self._clip.labels(names, MAKE_PROMPTS, k))
                       for k, names in makes.items()}
        self._colours = {k: self._clip.labels(config.COLOURS, [COLOUR_PROMPT], k) for k in makes}

    def find(self, frame, plate_box):
        """Box and type of the vehicle that carries the plate, or (None, None)."""
        cx, cy = (plate_box[0] + plate_box[2]) / 2, (plate_box[1] + plate_box[3]) / 2
        best = None
        for d in self._detector.predict(frame):
            if d.label not in config.VEHICLE_TYPES:
                continue
            b = d.bounding_box
            if b.x1 <= cx <= b.x2 and b.y1 <= cy <= b.y2:
                area = (b.x2 - b.x1) * (b.y2 - b.y1)
                if best is None or area < best[0]:       # innermost vehicle wins
                    best = (area, (max(0, b.x1), max(0, b.y1), b.x2, b.y2), d.label)
        return (best[1], best[2]) if best else (None, None)

    def describe(self, frame, plate_box):
        box, kind = self.find(frame, plate_box)
        if box is None:
            return Vehicle()
        x1, y1, x2, y2 = box
        emb = self._clip.embed_image(frame[y1:y2, x1:x2])
        colour, _ = _pick(config.COLOURS, self._colours[kind], emb)
        make, conf = _pick(*self._makes[kind], emb)
        return Vehicle(box, kind, colour, make if conf >= config.MAKE_MIN_CONF else None, conf)
