# Automated Quality Inspection System for Manufacturing (AQISM)

A computer vision prototype for detecting defects on manufactured items (specifically PCBs) using a reference-based comparison approach.

## Overview

This system identifies defects such as **scratches**, **missing components**, and **discoloration** by comparing a test image against a reliable "Golden Master" reference image. It utilizes OpenCV for image alignment (homography) and difference analysis.

## Features

- **Image Alignment**: Automatically aligns test images to the reference using ORB feature matching, making it robust to slight position/rotation shifts.
- **Defect Detection**:
  - **Scratches**: Detected via structural differences.
  - **Missing Components**: Detected via value/intensity differences in specific ROIs.
  - **Discoloration**: Detected via Hue shifts in HSV color space.
- **Reporting**: Outputs visual bounding boxes and confidence scores for each defect.

## Project Structure

```
├── dataset/             # Contains the Golden Master and test images
├── output/              # Generated result images with annotations
├── src/
│   ├── dataset_generator.py  # Script to generate synthetic defective images
│   └── inspector.py          # Core inspection logic class
├── main.py              # Entry point script
├── requirements.txt     # Python dependencies
└── README.md
```

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Generate Dataset** (Optional if you want to regenerate test images):
   ```bash
   python src/dataset_generator.py
   ```
   This will create synthetic test images in the `dataset/` folder.

2. **Run Inspection**:
   ```bash
   python main.py
   ```
   The script will process all images in `dataset/test_*.png`, detect defects, and save the annotated results to `output/`.

## Logic

1. **Registration**: The input image is aligned to `golden_master.png` to ensure pixel-perfect comparison.
2. **Difference Calculation**: Absolute difference is computed.
3. **Thresholding & Morphology**: Noise is removed, and significant differences are isolated as contours.
4. **Classification**: Each contour is analyzed (color, shape, intensity) to classify the defect type.
