import os
import tempfile

import re
import sys
from collections import Counter, defaultdict

import cv2
import imutils
import numpy as np
import pytesseract

MIN_AREA, MAX_AREA = 4500, 30000
MIN_RATIO, MAX_RATIO = 3, 6
MAX_TILT = 15

WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
DEFAULT_IMAGE = "src/testset/car.jpg"

TARGET_H = 160   # candidate crops are resized to this height for locating text
TEXT_H = 60      # character height (px) handed to Tesseract
MIN_CHARS = 3
DEBUG = False    # True prints every OCR pass


# ---------- Detection (steps 1-6) ----------
def normalize(rect):
    (cx, cy), (w, h), angle = rect
    if w < h:
        w, h = h, w
        angle += 90
    angle = ((angle + 90) % 180) - 90
    return (cx, cy), (w, h), angle


def validate(rect):
    _, (w, h), angle = normalize(rect)
    if w == 0 or h == 0:
        return False
    return (
        MIN_AREA <= w * h <= MAX_AREA
        and MIN_RATIO <= w / h <= MAX_RATIO
        and abs(angle) <= MAX_TILT
    )


def detect_candidates(img):
    gray = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2GRAY)

    sobel_x = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    sobel_x = np.uint8(255 * sobel_x / max(sobel_x.max(), 1))

    _, binary = cv2.threshold(sobel_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    cnts = imutils.grab_contours(
        cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    )
    return [r for r in map(cv2.minAreaRect, cnts) if validate(r)]


# ---------- Cropping ----------
def crop_rotated(img, rect, pad_x=0.08, pad_y=0.25):
    """Level the rectangle and cut it out with generous padding."""
    (cx, cy), (w, h), angle = normalize(rect)
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    size = (int(w * (1 + 2 * pad_x)), int(h * (1 + 2 * pad_y)))
    return cv2.getRectSubPix(rotated, size, (cx, cy))


def group_chars(binary):
    """Find character-like components and group them into rows."""
    H, _ = binary.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (0.25 * H <= h <= 0.95 * H):    # frame too tall, screws too small
            continue
        if not (0.08 * h <= w <= 1.1 * h):     # allows thin "1" up to wide "W"
            continue
        if area < 0.15 * w * h:
            continue
        comps.append((x, y, w, h))
    if len(comps) < MIN_CHARS:
        return None

    # Keep components with a similar height
    med = np.median([c[3] for c in comps])
    comps = [c for c in comps if 0.75 * med <= c[3] <= 1.25 * med]
    if len(comps) < MIN_CHARS:
        return None

    # Group into rows by vertical centre (handles two-line plates)
    comps.sort(key=lambda c: c[1] + c[3] / 2)
    rows = [[comps[0]]]
    for c in comps[1:]:
        cy = c[1] + c[3] / 2
        row_cy = np.mean([r[1] + r[3] / 2 for r in rows[-1]])
        if cy - row_cy > 0.5 * med:
            rows.append([c])
        else:
            rows[-1].append(c)
    rows = [r for r in rows if len(r) >= 2]
    kept = [c for r in rows for c in r]
    if len(kept) < MIN_CHARS:
        return None

    hs = np.array([c[3] for c in kept], dtype=float)
    uniform = 1 - min(hs.std() / hs.mean(), 0.5)
    return {
        "box": (
            min(c[0] for c in kept),
            min(c[1] for c in kept),
            max(c[0] + c[2] for c in kept),
            max(c[1] + c[3] for c in kept),
        ),
        "n": len(kept),
        "rows": len(rows),
        "char_h": float(np.median(hs)),
        "score": len(kept) * uniform,
    }


def locate_text(crop):
    """Tight crop around the characters of one candidate, or None."""
    if crop.shape[0] < 10 or crop.shape[1] < 10:
        return None

    scale = TARGET_H / crop.shape[0]
    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    soft = cv2.GaussianBlur(gray, (3, 3), 0)
    H, W = gray.shape
    block = (H // 2) | 1

    best = None
    for pol in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):  # dark-on-light, light-on-dark
        _, otsu = cv2.threshold(soft, 0, 255, pol + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(
            soft, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, pol, block, 10
        )
        for binary in (otsu, adaptive):
            res = group_chars(binary)
            if res and (best is None or res["score"] > best["score"]):
                best = res
    if best is None:
        return None

    x0, y0, x1, y1 = best["box"]
    m = int(0.15 * (y1 - y0))
    best["gray"] = gray[max(0, y0 - m):min(H, y1 + m), max(0, x0 - m):min(W, x1 + m)]
    return best


def find_best_plate(img):
    """Score every candidate and return only the best one."""
    best = None
    for rect in detect_candidates(img):
        crop = crop_rotated(img, rect)
        found = locate_text(crop)
        if found and (best is None or found["score"] > best["score"]):
            found["rect"] = rect
            found["crop"] = crop
            best = found
    return best


# ---------- OCR ----------
def run_ocr(img, psm):
    cfg = f"--oem 1 --psm {psm} -c tessedit_char_whitelist={WHITELIST}"
    data = pytesseract.image_to_data(
        img, config=cfg, output_type=pytesseract.Output.DICT
    )
    words, confs = [], []
    for t, c in zip(data["text"], data["conf"]):
        t = re.sub(r"[^A-Za-z0-9]", "", t).upper()
        if t and float(c) >= 0:
            words.append(t)
            confs.append(float(c))
    if not words:
        return "", 0.0
    return "".join(words), sum(confs) / len(confs)


def read_plate(found):
    gray = found["gray"]
    scale = TEXT_H / found["char_h"]
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interp)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)

    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(otsu) < 127:
        otsu = cv2.bitwise_not(otsu)

    adaptive = cv2.adaptiveThreshold(
        clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        (int(TEXT_H * 0.75)) | 1, 10,
    )
    if np.mean(adaptive) < 127:
        adaptive = cv2.bitwise_not(adaptive)

    variants = {"clahe": clahe, "otsu": otsu, "adaptive": adaptive}
    psm = 7 if found["rows"] == 1 else 6

    results = []
    for name, v in variants.items():
        v = cv2.copyMakeBorder(v, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
        text, conf = run_ocr(v, psm)
        if DEBUG:
            print(f"  {name:9s} psm{psm} -> {text!r} ({conf:.0f})")
        if len(text) >= MIN_CHARS:
            results.append((text, conf))
    if not results:
        return ""

    # One answer: per-position weighted vote among results of the modal length
    length = Counter(len(t) for t, _ in results).most_common(1)[0][0]
    same = [(t, c) for t, c in results if len(t) == length]
    best = ""
    for i in range(length):
        score = defaultdict(float)
        for t, c in same:
            score[t[i]] += c
        best += max(score, key=score.get)
    return best


# ---------- Entry point ----------
def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else DEFAULT_IMAGE
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read {path}, check the path")

    found = find_best_plate(img)
    if found is None:
        print("No plate found")
        return

    text = read_plate(found)
    print(f"Plate: {text!r}  ({found['n']} characters, {found['rows']} row(s))")

    annotated = img.copy()
    box = cv2.boxPoints(found["rect"]).astype(int)
    cv2.drawContours(annotated, [box], 0, (0, 255, 0), 2)
    cv2.putText(annotated, text, (box[:, 0].min(), max(box[:, 1].min() - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show Window
    # cv2.imshow("Plate text", found["gray"])
    cv2.imshow("Result", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optional: save debug images to the system temp dir when asked
    if "--debug" in sys.argv:
        out = tempfile.mkdtemp(prefix="license_")
        cv2.imwrite(os.path.join(out, "plate_crop.png"), found["crop"])
        cv2.imwrite(os.path.join(out, "plate_text.png"), found["gray"])
        cv2.imwrite(os.path.join(out, "result.png"), annotated)
        print(f"Debug images saved to {out}")


if __name__ == "__main__":
    main()
