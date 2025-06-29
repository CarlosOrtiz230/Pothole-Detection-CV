import numpy as np
from skimage.transform import ProjectiveTransform, warp
from utils.preprocess import to_gray

def detect(img):
    """Detect potholes in angled images using a perspective 'flattening'.
    Returns:
        confidence (float): darkness in region of interest
        box (tuple): (x, y, w, h)
    """
    gray = to_gray(img)
    h, w = gray.shape

    # Step 1: Define a rough "perspective-flattening" homography
    src = np.array([
        [w * 0.2, h * 0.6],
        [w * 0.8, h * 0.6],
        [w * 0.95, h * 0.95],
        [w * 0.05, h * 0.95],
    ])

    dst = np.array([
        [w * 0.25, h * 0.25],
        [w * 0.75, h * 0.25],
        [w * 0.75, h * 0.75],
        [w * 0.25, h * 0.75],
    ])

    transform = ProjectiveTransform()
    transform.estimate(src, dst)

    warped = warp(gray, transform, output_shape=(h, w))
    warped = (warped * 255).astype(np.uint8)

    # Step 2: Take central region (flattened) and measure darkness
    cx, cy = w // 2, h // 2
    region = warped[cy-30:cy+30, cx-40:cx+40]
    dark_mask = region < 60
    confidence = dark_mask.mean()

    box = (cx - 40, cy - 30, 80, 60)

    return confidence, box
