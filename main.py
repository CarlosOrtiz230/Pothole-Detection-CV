#!/usr/bin/env python3
# Main script for unified pothole detection CLI

import sys
import random
from pathlib import Path
import cv2
import numpy as np

from detector import detect_best_pothole
from utils.draw import draw_box
from utils.preprocess import to_gray

SAMPLES_DIR = Path(__file__).parent / "samples"

# List all valid image files from the samples directory
def list_images():
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(p for p in SAMPLES_DIR.glob("*") if p.suffix.lower() in exts)

# Load image using OpenCV and convert BGR to RGB
def load_image(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise ValueError(f"Failed to load image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Display image using OpenCV
def show_image(img: np.ndarray):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Detection Result", img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run detection on a single image
def run_on_image(path, debug=False):
    img = load_image(path)
    score, box = detect_best_pothole(img, debug=debug)
    print(f"\n{path.name} → confidence: {score:.2f}")
    if score >= 0.5:
        print(" POTHOLE DETECTED!")
        img_marked = draw_box(img.copy(), box, color=(255, 0, 0))
        show_image(img_marked)
    else:
        print(" No pothole detected.")
        show_image(img)

# Run detection over multiple images
def run_batch(paths, debug=False):
    print(f"\nRunning on {len(paths)} images...\n")
    for path in paths:
        run_on_image(path, debug=debug)

# Entry point
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
        print("\nAvailable images:")
        for i, path in enumerate(paths, 1):
            print(f"  {i}. {path.name}")
        try:
            choice = int(input("Choose image [1-n]: "))
            run_on_image(paths[choice - 1], debug=debug)
        except:
            sys.exit("Invalid image selection.")
    elif mode == 2:
        subset = random.sample(paths, min(5, len(paths)))
        run_batch(subset, debug=debug)
    elif mode == 3:
        run_batch(paths, debug=debug)
    else:
        sys.exit("Invalid mode selected.")

if __name__ == "__main__":
    main()
