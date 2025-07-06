import cv2
import numpy as np

def to_gray(img):
    """
    Converts an RGB or BGR image to grayscale.
    """
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img.copy()
