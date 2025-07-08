import numpy as np
import cv2
from utils.preprocess import to_gray, preprocess_with_otsu_and_erosion
from features import (
    get_darkness_score,
    get_texture_score,
    get_shape_score,
    get_orb_score
)

CFG = dict(
    blur_kernel        = (5, 5),
    adaptive_blocksize = 21,
    adaptive_C         = 10,
    min_area_px        = 150,
    confidence_thresh  = 0.5,
    debug              = False
)

def detect_best_pothole(img, cfg=None, debug=False):
    p = {**CFG, **(cfg or {})}
    gray = to_gray(img)

    results = []

    # --- OTSU path ---
    mask_otsu = preprocess_with_otsu_and_erosion(gray)
    results.append(_get_best_from_mask(mask_otsu, img, gray, p, debug, label="OTSU"))

    # --- Adaptive Threshold path ---
    blurred = cv2.GaussianBlur(gray, p['blur_kernel'], 0)
    mask_adapt = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        p['adaptive_blocksize'],
        p['adaptive_C']
    )
    results.append(_get_best_from_mask(mask_adapt, img, gray, p, debug, label="ADAPTIVE"))

    # --- Return best of both ---
    best_conf, best_box = max(results, key=lambda x: x[0])
    if best_conf >= p['confidence_thresh']:
        return best_conf, best_box
    return 0.0, (0, 0, 1, 1)

# --- Helper for extracting best candidate from a binary mask ---
def _get_best_from_mask(binary, img, gray, p, debug=False, label=""):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if debug:
        print(f"[{label}] Found {len(contours)} contours")

    best_conf = 0.0
    best_box = None

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < p['min_area_px']:
            continue

        roi = gray[y:y+h, x:x+w]
        c_dark   = get_darkness_score(roi, debug)
        c_tex    = get_texture_score(roi, debug)
        c_shape  = get_shape_score(cnt, debug)
        c_orb    = get_orb_score(img, (x, y, w, h), debug)

        confidence = (
            0.3 * c_dark +
            0.3 * c_tex +
            0.2 * c_shape +
            0.2 * c_orb
        )

        if debug:
            print(f"[{label}] CANDIDATE @({x},{y},{w},{h}) → conf: {confidence:.2f}")

        if confidence > best_conf:
            best_conf = confidence
            best_box = (x, y, w, h)

    return best_conf, best_box
