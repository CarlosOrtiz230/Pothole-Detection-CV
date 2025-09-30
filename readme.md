# Pothole Detection with Classical Computer Vision and YOLO

This project provides **two main functionalities** for pothole detection:

1. **Classical Computer Vision (no deep learning)**  
   Detect potholes in road images using traditional image processing techniques such as intensity transformations, filtering, segmentation, and feature extraction.  
   - Useful for lightweight solutions or as a first step to prepare training datasets.  
   - Robust to different real-world scenarios:
     - Centered and well-lit potholes
     - Displaced or partially visible
     - Seen in angled perspective
     - Far away or low-resolution
     - With distracting elements like shadows, tires, bags, etc.

2. **YOLO Inference Model (deep learning)**  
   Detect potholes using a trained YOLO model for real-time and more scalable applications.  
   - Suitable for automated detection pipelines.  
   - Can be extended with different YOLO versions (v5, v7, v8, etc.) depending on resource constraints.  

## Objective

- Build modular Python scripts that allow:
  - Running **classical CV-based pothole detection**.  
  - Running **YOLO inference** on road images or videos.  
  - Comparing performance, speed, and accuracy between both approaches.
