import numpy as np
import cv2
from utils.preprocess import to_gray, preprocess_with_otsu_and_erosion
from features import get_orb_score


CFG = dict(
    min_area_px       = 100,
    confidence_thresh = 0.5,
)

def detect_best_pothole(img, cfg=None, debug=False):
    p = {**CFG, **(cfg or {})}
    gray = to_gray(img)

    # Step-by-step preprocessing
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(otsu, kernel, iterations=1)

    if debug:
        visualize_pipeline(img, gray, enhanced, otsu, eroded)

    # --- Detection ---
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if debug:
        print(f"[DEBUG - EASY] Found {len(contours)} contours")

    best_conf = 0.0
    best_box = None

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < p['min_area_px']:
            continue

        score = get_orb_score(img, (x, y, w, h), debug)

        if debug:
            print(f"[EASY CANDIDATE] @({x},{y},{w},{h}) → ORB score: {score:.2f}")

        if score > best_conf:
            best_conf = score
            best_box = (x, y, w, h)

    if best_conf >= p['confidence_thresh'] and best_box:
        return best_conf, best_box
    return 0.0, (0, 0, 1, 1)

# --- Multi-view Visualization Function ---
def visualize_pipeline(rgb, gray, enhanced, otsu, eroded):
    def resize(img):
        return cv2.resize(img, (256, 256))

    # Convert grayscale images to BGR for display
    gray_bgr    = cv2.cvtColor(resize(gray), cv2.COLOR_GRAY2BGR)
    enh_bgr     = cv2.cvtColor(resize(enhanced), cv2.COLOR_GRAY2BGR)
    otsu_bgr    = cv2.cvtColor(resize(otsu), cv2.COLOR_GRAY2BGR)
    eroded_bgr  = cv2.cvtColor(resize(eroded), cv2.COLOR_GRAY2BGR)
    rgb_resized = resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    combined = np.hstack([rgb_resized, gray_bgr, enh_bgr, otsu_bgr, eroded_bgr])
    cv2.imshow("Easy Pipeline: RGB | Gray | CLAHE | Otsu | Eroded", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
