import sys
import random
from pathlib import Path
import cv2
import numpy as np
import argparse

from detector import detect_best_pothole
from utils.draw import draw_box
from utils.preprocess import to_gray, load_and_resize

ROOT_SAMPLES_DIR = Path(__file__).parent / "samples"

LABEL_MAP = {
    "potholes_samples": 1,
    "clean_samples": 0
}

def list_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in exts)

def load_image(path: Path) -> np.ndarray:
    return load_and_resize(path)

def show_image(img: np.ndarray):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Detection Result", img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def clean_filename(path: Path) -> str:
    name = path.stem
    if "jpg.rf." in name:
        name = name.split("jpg.rf.")[-1][:12]
    return name + path.suffix

def detect_and_check(path: Path, true_label: int, debug=False, show=False):
    img = load_image(path)
    conf, box = detect_best_pothole(img, debug=debug)
    predicted = int(conf >= 0.5)
    correct = (predicted == true_label)
    label_name = "POTHOLE" if predicted else "CLEAN"
    pretty_name = clean_filename(path)

    print(f"\n{pretty_name} → confidence: {conf:.2f} → predicted: {label_name} → {'✔️' if correct else '❌'}")

    if show:
        img_marked = draw_box(img.copy(), box, color=(255, 0, 0)) if predicted else img
        show_image(img_marked)

    return correct

def hybrid_test(num_each=5, debug=False, show=False):
    folders = ["potholes_samples", "clean_samples"]
    all_paths = []

    for label_name in folders:
        folder = ROOT_SAMPLES_DIR / label_name
        images = list_images(folder)
        sampled = random.sample(images, min(num_each, len(images)))
        all_paths.extend((p, LABEL_MAP[label_name]) for p in sampled)

    random.shuffle(all_paths)

    total = len(all_paths)
    correct = 0

    for path, label in all_paths:
        if detect_and_check(path, label, debug=debug, show=show):
            correct += 1

    print(f"\n[SUMMARY] Correct: {correct}/{total} ({100*correct/total:.1f}%)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testing", action="store_true", help="Run quick hybrid test with 3+3 images")
    args = parser.parse_args()

    if args.testing:
        hybrid_test(num_each=3, debug=True, show=True)
        return

    print("\nSelect test mode:")
    print("  1. One image (manual)")
    print("  2. Random clean samples")
    print("  3. Random pothole samples")
    print("  4. Hybrid test (e.g. 5+5, with accuracy)")
    try:
        choice = int(input("Select mode [1-4]: "))
    except:
        sys.exit("Invalid mode selection.")

    debug = input("Enable debug mode? (y/n): ").strip().lower() == "y"
    show = input("Show images? (y/n): ").strip().lower() == "y"

    if choice == 1:
        print("\nChoose folder:")
        print("  1. potholes_samples")
        print("  2. clean_samples")
        d = int(input("Select [1-2]: "))
        subfolder = "potholes_samples" if d == 1 else "clean_samples"
        label = LABEL_MAP[subfolder]
        folder = ROOT_SAMPLES_DIR / subfolder
        images = list_images(folder)
        print("\nAvailable images:")
        for i, p in enumerate(images, 1):
            print(f"  {i}. {clean_filename(p)}")
        idx = int(input("Select image [1-n]: "))
        detect_and_check(images[idx - 1], label, debug=debug, show=True)

    elif choice in {2, 3}:
        subfolder = "clean_samples" if choice == 2 else "potholes_samples"
        label = LABEL_MAP[subfolder]
        folder = ROOT_SAMPLES_DIR / subfolder
        images = list_images(folder)
        k = int(input("How many images to test?: "))
        subset = random.sample(images, min(k, len(images)))
        correct = 0
        for p in subset:
            if detect_and_check(p, label, debug=debug, show=show):
                correct += 1
        print(f"\n[SUMMARY] Correct: {correct}/{len(subset)} ({100*correct/len(subset):.1f}%)")

    elif choice == 4:
        n = int(input("How many from each category?: "))
        hybrid_test(num_each=n, debug=debug, show=show)

    else:
        sys.exit("Invalid mode.")

if __name__ == "__main__":
    main()
