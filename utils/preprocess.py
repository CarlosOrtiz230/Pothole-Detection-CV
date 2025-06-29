import numpy as np
from PIL import Image

def resize(img: np.ndarray, scale: float) -> np.ndarray:
    """Resize image by scale factor using PIL."""
    h, w = img.shape[:2]
    new_size = (int(w * scale), int(h * scale))
    return np.asarray(Image.fromarray(img).resize(new_size))


def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert RGB image (H x W x 3) to grayscale (H x W) using average method."""
    if img.ndim == 3 and img.shape[2] == 3:
        return img.mean(axis=2).astype(np.uint8)
    raise ValueError("Expected RGB image with shape (H, W, 3)")
