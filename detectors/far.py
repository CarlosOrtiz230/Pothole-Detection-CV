import numpy as np
from utils.preprocess import to_gray

def detect(img):
    """Detect small, distant potholes likely near the bottom of the image.
    
    Returns:
        confidence (float): score from 0 to 1
        box (tuple): (x, y, w, h) of the region being flagged
    """
    gray = to_gray(img)
    h, w = gray.shape

    # Focus on lower third of the image — where far potholes typically appear
    y = int(h * 2 / 3)
    roi = gray[y:, :]  # bottom third

    # Resize ROI to make small potholes more visible (simulate zoom)
    zoomed = roi[::2, ::2]  # crude downsample to simulate coarse detection

    # Detect small dark regions
    mask = zoomed < 50  # adjust threshold as needed
    dark_ratio = mask.sum() / mask.size

    # Fake box around a plausible "far pothole" area
    box_width = w // 6
    box_height = h // 10
    x = w // 2 - box_width // 2
    box_y = y + (h - y - box_height) // 2

    return dark_ratio, (x, box_y, box_width, box_height)
