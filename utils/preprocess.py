import cv2
import numpy as np
from pathlib import Path

def to_gray(img):
    """
    Converts an RGB or BGR image to grayscale.
    """
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img.copy()

def load_and_resize(path: Path, size=(512, 512)) -> np.ndarray:
    """
    Loads an image from disk, converts it to RGB, and resizes it.
    Args:
        path (Path): Path to image file.
        size (tuple): Target size in (width, height).
    Returns:
        np.ndarray: Resized RGB image.
    """
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise ValueError(f"Could not load image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, size, interpolation=cv2.INTER_AREA)
    return img_rgb

def preprocess_with_otsu_and_erosion(gray):
    # Apply CLAHE first
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Global OTSU threshold
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Erosion with 3x3 kernel
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)

    return eroded
