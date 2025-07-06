import numpy as np
import cv2

# --- Darkness Score ---
def get_darkness_score(roi, debug=False):
    mean = roi.mean()
    threshold = mean * 0.6
    mask = roi < threshold
    dark_frac = mask.mean()

    if debug:
        print(f"[DARKNESS] ROI mean={mean:.1f}, threshold={threshold:.1f}, dark_fraction={dark_frac:.3f}")

    return float(np.clip(dark_frac * 2.5, 0.0, 1.0))  # Boost scoring

# --- Texture Score (standard deviation) ---
def get_texture_score(roi, debug=False):
    sigma = roi.std()

    # Normalize std dev to score between 0 and 1
    score = np.clip(sigma / 25.0, 0.0, 1.0)

    if debug:
        print(f"[TEXTURE] σ={sigma:.2f}, score={score:.2f}")

    return float(score)

# --- Shape Score ---
def get_shape_score(contour, debug=False):
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    rect_area = w * h
    aspect = w / (h + 1e-6)

    extent = area / (rect_area + 1e-6)
    circularity = 4 * np.pi * area / (cv2.arcLength(contour, True)**2 + 1e-6)

    score = 0.0
    if 0.5 < aspect < 2.0 and extent > 0.4:
        score = np.clip(circularity * 2, 0.0, 1.0)

    if debug:
        print(f"[SHAPE] aspect={aspect:.2f}, extent={extent:.2f}, circularity={circularity:.2f}, score={score:.2f}")

    return float(score)

# --- ORB Keypoint Density ---
def get_orb_score(img, box, debug=False):
    x, y, w, h = box
    roi_gray = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create()
    keypoints = orb.detect(roi_gray, None)

    density = len(keypoints) / (w * h + 1e-6)
    score = np.clip(density * 3000, 0.0, 1.0)

    if debug:
        print(f"[ORB] keypoints={len(keypoints)}, density={density:.5f}, score={score:.2f}")

    return float(score)
