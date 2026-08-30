# 👁️ OD-Edge-SAM: Optic Disc Edge Segment Anything Model

**Official Open-Source Release for the Paper:**
> **"OD-Edge-SAM: Optic Disc Edge Segment Anything Model for Eye Fundus Screening via Adaptive Image Abstraction on Embedded Processors"**
>
> *Juan Herón Rodríguez-Reséndiz & Jorge Francisco Martínez-Carballido (2026)*

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/nc-nd/4.0/)
[![Runtime: ONNX](https://img.shields.io/badge/Runtime-ONNX%20v1.16+-blue.svg)](#)
[![Hardware: Embedded CPU](https://img.shields.io/badge/Target-ARM%20Cortex--A76%20%2F%20Raspberry%20Pi%205-red.svg)](#)

---

## 📌 Overview

This repository provides the standalone abstraction executable and the exported **ONNX models** for **OD-Edge-SAM**, an edge‑deployable framework for zero-shot Optic Disc (OD) segmentation on low-power embedded processors (such as the Raspberry Pi 5).

By combining an **Adaptive Histogram-Guided Image Abstraction Layer** (`abstraction.exe`) with an edge-adapted Segment Anything Model 2.1 (SAM 2.1) Tiny architecture (`OD_Edge_SAM_encoder.onnx` and `OD_Edge_SAM_decoder.onnx`), this pipeline enables fully autonomous segmentation without manual prompting or dataset‑specific retraining.

---

## 📖 Citation

If you use this repository, the abstraction executable, or the exported ONNX models in your research, please cite our manuscript:

```bibtex
@article{Rodriguez2026ODEdgeSAM,
  title={OD-Edge-SAM: Optic Disc Edge Segment Anything Model for Eye Fundus Screening via Adaptive Image Abstraction on Embedded Processors},
  author={Rodríguez-Reséndiz, Juan Herón and Martínez-Carballido, Jorge Francisco},
  journal={Submission Pending},
  year={2026}
}
```
## ⚠️ Important: Input Image Cropping Requirement
>[!IMPORTANT]
>OD-Edge-SAM requires Region-of-Interest (ROI) cropped images and **CANNOT** be executed directly on full-view retinal fundus photographs.

* **Required Input Region:** Each input image must be tightly cropped around the Optic Disc and its immediate Peripapillary Atrophy (PPA) margin.

* **Why Full-Field Images Fail:** The abstraction algorithm computes 1D intensity histograms to detect dominant tissue clusters. Full-field fundus photographs introduce macular tissue, background pigmentation, and illumination gradients that distort peak detection, leading to unstable clustering and incorrect prompt extraction.

## 📂 Repository Structure
```text
├── abstraction.exe               # Standalone binary for adaptive histogram K-Means abstraction
├── OD_Edge_SAM_encoder.onnx      # Exported Hiera-Tiny Image Encoder (ONNX format)
├── OD_Edge_SAM_decoder.onnx      # Exported Mask Decoder & Prompt Engine (ONNX format)
├── LICENSE                       # CC BY 4.0 Legal Text
└── README.md                     # Documentation file
```
## 🛠️ Standalone Abstraction Engine (abstraction.exe)

`abstraction.exe`  is a standalone Windows binary (no Python environment required). It pre‑conditions cropped fundus images by isolating intensity clusters in the Red (R) and Green (G) channels while suppressing background noise.

### Command-Line Usage

```bash
abstraction.exe <input_crop_path> <output_abstracted_path> <output_hist_path> [--mode {BASE,CONS,SENS}]
```

### Modes & Sensitivity Settings
* **BASE (Default):** Standard peak detection balance ($div=6.0$, $dist_R=25$, $dist_G=11$) suited for standard clinical crops.
* **CONS (Conservative):** Strong peak separation ($div=4.0$, $dist_R=40$, $dist_G=20$) to avoid oversegmentation on high-contrast margins.
* **SENS (Sensitive):** Detects subtle low-amplitude peaks ($div=10.0$, $dist_R=10$, $dist_G=5$) in low-contrast or hazy fundus crops.

### Execution Examples
```bash
# Default mode (BASE)
abstraction.exe "images/crop/sample.jpg" "images/abstracted/sample.jpg" "images/hist/sample.jpg"

# Conservative mode (CONS)
abstraction.exe "images/crop/sample.jpg" "images/abstracted/samplecons.jpg" "images/hist/samplecons.jpg" --mode CONS

# Sensitive mode (SENS)
abstraction.exe "images/crop/sample.jpg" "images/abstracted/samplesens.jpg" "images/hist/samplesens.jpg" --mode SENS
```
## 🎯 Autonomous Prompting & ONNX Inference Pipeline

The system segments the optic disc autonomously using a brightest‑point prompting mechanism:

1. Autonomous Prompt Localization: The optic cup and neuroretinal rim appear brightest in the cropped region. The script identifies the brightest coordinate  (`maxLoc`) within the central 15% radius ROI of the abstracted image.
2. Coordinate Scaling: The detected point is mapped to SAM’s $1024\times1024$ input space and passed as a single positive point prompt $[\hat{x}, \hat{y}]$ to the Mask Decoder.
3. Direct Disc Extraction: The resulting mask is applied to the original crop using (`cv2.bitwise_and`) to obtain the segmented optic disc.

### Python Inference Script (`run_inference.py`)

```python
"""
OD-Edge-SAM: Autonomous Optic Disc Segmentation via ONNX Runtime
"""
import os
import cv2
import numpy as np
import onnxruntime as ort

# 1. Load ONNX Runtime Sessions
encoder_session = ort.InferenceSession("OD_Edge_SAM_encoder.onnx", providers=['CPUExecutionProvider'])
decoder_session = ort.InferenceSession("OD_Edge_SAM_decoder.onnx", providers=['CPUExecutionProvider'])

# 2. Load Raw Crop and Abstracted Images
crop_img_path = "images/crop/sample.jpg"
abstracted_img_path = "images/abstracted/sample.jpg"

orig_crop = cv2.imread(crop_img_path)
abstracted_img = cv2.imread(abstracted_img_path)
if orig_crop is None or abstracted_img is None:
    raise FileNotFoundError("Input images could not be loaded. Please check the provided paths.")

H_orig, W_orig = orig_crop.shape[:2]

# 3. Autonomous Prompt Extraction (Brightest Point in Central ROI)
grey_abstracted = cv2.cvtColor(abstracted_img, cv2.COLOR_BGR2GRAY)
mask_center = np.zeros_like(grey_abstracted, dtype=np.uint8)
cv2.circle(mask_center, (W_orig // 2, H_orig // 2), int(W_orig * 0.15), 255, -1)
_, _, _, maxLoc = cv2.minMaxLoc(grey_abstracted, mask=mask_center)

# Scale coordinates to the 1024x1024 input space
pt_x_1024 = float(maxLoc[0] * (1024.0 / W_orig))
pt_y_1024 = float(maxLoc[1] * (1024.0 / H_orig))
point_coords = np.array([[[pt_x_1024, pt_y_1024]]], dtype=np.float32)
point_labels = np.array([[1.0]], dtype=np.float32)

# 4. Preprocess Abstracted Image (1024x1024, ImageNet Normalization)
img_rgb = cv2.cvtColor(abstracted_img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (1024, 1024)).astype(np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
tensor_input = ((img_resized - mean) / std).transpose(2, 0, 1)[np.newaxis, :, :, :]

# 5. Image Encoder Forward Pass
enc_inputs = {encoder_session.get_inputs()[0].name: tensor_input}
enc_outputs = encoder_session.run(None, enc_inputs)
high_res_0, high_res_1, image_embed = enc_outputs[0], enc_outputs[1], enc_outputs[2]

# 6. Mask Decoder Forward Pass with Autonomous Prompt
dense_pe = np.zeros((1, 256, 64, 64), dtype=np.float32)
dec_inputs = {
    "image_embeddings": image_embed,
    "image_pe": dense_pe,
    "point_coords": point_coords,
    "point_labels": point_labels,
    "high_res_feat_0": high_res_0,
    "high_res_feat_1": high_res_1
}
low_res_masks, scores = decoder_session.run(None, dec_inputs)
best_mask = low_res_masks[0, int(np.argmax(scores[0]))]

# 7. Generate Segmented Optic Disc Image
mask_resized = cv2.resize(best_mask, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
binary_mask = (mask_resized > 0.0).astype(np.uint8) * 255

segmented_disc = cv2.bitwise_and(orig_crop, orig_crop, mask=binary_mask)

cv2.imwrite("output_disc.png", segmented_disc)
cv2.imwrite("output_mask.png", binary_mask)
print(f"[SUCCESS] Segmentation finished. Autonomous prompt placed at crop coordinate: ({maxLoc[0]}, {maxLoc[1]})")
```

## 📜 License

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/nc-nd/4.0/)

This project, including the executable binary (`abstraction.exe`) and exported ONNX models (`OD_Edge_SAM_encoder.onnx`, `OD_Edge_SAM_decoder.onnx`), is licensed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)** license.

**Under the following terms:**
* **NonCommercial:** You may not use the material for commercial purposes.
* **NoDerivatives:** If you remix, transform, or build upon the material, you may not distribute the modified material.
* **Attribution:** Appropriate credit and paper citation are mandatory.
