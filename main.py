#!/usr/bin/env python3

import sys
import random
from pathlib import Path
import cv2
import numpy as np

from detector import detect_best_pothole  # unified detection function
from utils.draw import draw_box
from utils.preprocess import to_gray

# Directory with test images
SAMPLES_DIR = Path(__file__).parent / "samples"

# List image files
def list_images():
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(p for p in SAMPLES_DIR.glob("*") if p.suffix.lower() in exts)

# Load image and convert to RGB
def load_image(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise ValueError(f"Could not load image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Show image with OpenCV
def show_image(img_rgb: np.ndarray, title="Detection Result"):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow(title, img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run detection on one image
def run_on_image(path: Path, debug=False):
    img = load_image(path)
    confidence, box = detect_best_pothole(img, debug=debug)

    print(f"\n{path.name} → confidence: {confidence:.2f}")
    if confidence >= 0.5:
        print(" POTHOLE DETECTED!")
        img_boxed = draw_box(img.copy(), box, color=(255, 0, 0))
        show_image(img_boxed)
    else:
        print(" No pothole detected.")
        show_image(img)

# Run detection on a batch of images
def run_batch(paths, debug=False):
    print(f"\nRunning on {len(paths)} images...\n")
    for path in paths:
        run_on_image(path, debug=debug)

# CLI menu
def main():
    paths = list_images()
    if not paths:
        sys.exit("No images found in samples/")

    print("\nChoose mode:")
    print("  1. One image")
    print("  2. Random sample")
    print("  3. All images")
    try:
        mode = int(input("Select [1-3]: "))
    except:
        sys.exit("Invalid input")

    debug = input("Enable debug mode? (y/n): ").strip().lower() == "y"

    if mode == 1:
        print
