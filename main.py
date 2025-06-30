#!/usr/bin/env python3

import sys
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from detectors.centered import detect as detect_centered
from detectors.angled import detect as detect_angled
from detectors.far import detect as detect_far
from utils.draw import draw_box
from utils.preprocess import to_gray

# Folder with test images
SAMPLES_DIR = Path(__file__).parent / "samples"

# Register detectors
DETECTORS = {
    "centered": detect_centered,
    "angled": detect_angled,
    "far": detect_far,
}


def list_images():
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(p for p in SAMPLES_DIR.glob("*") if p.suffix.lower() in exts)


def load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def show_image(img: np.ndarray):
    Image.fromarray(img).show()


def run_on_image(path, detector_fn):
    img = load_image(path)
    score, box = detector_fn(img)  # detector returns confidence, box
    print(f"\n {path.name} → confidence: {score:.2f}")
    if score >= 0.5:
        print(" POTHOLE DETECTED!")
        img_marked = draw_box(img.copy(), box, color=(255, 0, 0))
        show_image(img_marked)
    else:
        print(" No pothole detected.")
        show_image(img)


def run_batch(paths, detector_fn):
    print(f"\n Running on {len(paths)} images...\n")
    for path in paths:
        run_on_image(path, detector_fn)


def main():
    # Choose analysis type
    print("\n Choose analysis type:")
    for idx, name in enumerate(DETECTORS.keys(), 1):
        print(f"  {idx}. {name}")
    try:
        d_choice = int(input("Select detector [1-3]: "))
        d_key = list(DETECTORS.keys())[d_choice - 1]
        detector_fn = DETECTORS[d_key]
    except:
        sys.exit("Invalid detector selection.")

    # Choose mode
    print("\n Choose mode:")
    print("  1. One image (manual pick)")
    print("  2. A few random samples")
    print("  3. All images")
    try:
        m_choice = int(input("Select mode [1-3]: "))
    except:
        sys.exit("Invalid mode selection.")

    paths = list_images()
    if not paths:
        sys.exit(" No images found in samples/")

    if m_choice == 1:
        # One image
        print("\n📷 Available images:")
        for idx, path in enumerate(paths, 1):
            print(f"  {idx}. {path.name}")
        try:
            i_choice = int(input("Choose image [1-n]: "))
            run_on_image(paths[i_choice - 1], detector_fn)
        except:
            sys.exit("Invalid image selection.")

    elif m_choice == 2:
        # Sample batch
        subset = random.sample(paths, min(5, len(paths)))
        run_batch(subset, detector_fn)

    elif m_choice == 3:
        # All images
        run_batch(paths, detector_fn)

    else:
        sys.exit("Invalid mode selected.")


if __name__ == "__main__":
    main()
