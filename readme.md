# Pothole Detection with Classical Computer Vision

This project aims to detect potholes in road images using **classical computer vision algorithms** without machine learning or deep learning — as a first step toward building a training dataset or deploying lightweight solutions.

## Objective

- Develop modular Python scripts to detect potholes under different real-world scenarios:
  - Centered and well-lit
  - Displaced or partially visible
  - Seen in angled perspective
  - Far away or low-resolution
  - With distracting elements like shadows, tires, bags, etc.

##  Project Structure

Pothole-Detection-CV/
├── main.py                # CLI: runs detector on images
├── requirements.txt       # pip packages
├── README.md              # Project overview

├── samples/               # 600 test images
│   └── *.jpg / *.png

├── detector.py            #  Unified detection logic: contour + scoring
│
├── features.py            # Feature extractors:
│   ├── get_darkness()
│   ├── get_texture()
│   ├── get_orb_density()
│   ├── check_shape(), etc.
│
└── utils/
    ├── preprocess.py      # to_gray(), blur, etc.
    └── draw.py            # draw_box(), overlay_confidence()
