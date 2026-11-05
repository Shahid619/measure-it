# Measure It — Optical Metric Measurement System

Automated industrial computer vision pipeline that takes a single photograph of a handheld object and reports its true physical width and height in millimetres using optical lens undistortion, deep learning semantic segmentation, and planar fiducial scale calibration.

---

## 1. System Overview & Architecture

The pipeline processes input images through five deterministic stages:

$$\text{Raw Image} \xrightarrow{\text{OpenCV Undistort}} \text{Metric Plane} \xrightarrow{\text{Reference Scale}} \text{Scale Ratio } (\text{mm/px}) \xrightarrow{\text{U-Net Mask}} \text{Oriented Bounding Box} \rightarrow \text{Measurements (mm)}$$

1. **Lens Undistortion:** Eliminates radial and tangential optical barrel distortion using precomputed intrinsic camera matrix $K$ and distortion coefficients $D$.
2. **Reference Scale Extraction:** Detects co-planar Sony Xperia 5 II ($158.0\text{ mm} \times 68.0\text{ mm}$) to determine physical millimetres-per-pixel ratio.
3. **Semantic Segmentation:** PyTorch Lightweight U-Net generates a high-confidence binary mask of the target card holder.
4. **Metric Dimension Extraction:** Fits minimum-area rotated bounding box (`cv2.minAreaRect`) and projects pixel dimensions to millimetres via the scale factor.
5. **Guardrail Rejection:** Rejects low-confidence masks, missing reference objects, or non-coplanar tilted scenes ($> 25^\circ$).

---

## 2. Hardware & Environment Specifications

- **OS:** Windows 10/11 x86_64
- **Python:** 3.12.8
- **Core Libraries:** OpenCV 5.0.0, PyTorch 2.9.0 (CPU), NumPy, Pandas
- **Capture Camera:** Smartphone camera (Locked 1.0x digital zoom, fixed focus, auto-exposure locked)
- **Image Resolution:** $1280 \times 960$ pixels
- **Model Checkpoint SHA-256:** `06d5cf1f0c3e5186e3c1edb5c4bbc88070fb1dbfdd11b7e49ec18f654582b68d` (`models/cardholder_unet.pth`)

---

## 3. Quickstart & Installation

```bash
# 1. Clone repository and checkout development branch
git clone <repo-url>
cd Demo
git checkout dev-test01

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 4. Single-Command Inference (`image-to-millimetres`)

Run the measurement pipeline on any image:
```bash
python -m inference.measure --image dataset/images/frame_000.png --output inference/output_measurement.png
```

### Sample Output:
```text
==================================================
MEASUREMENT PIPELINE RESULT:
  Status: SUCCESS
  Target Object: Card Holder
  Width (mm):    62.21 mm
  Height (mm):   94.28 mm
  Confidence:    0.8342
  Scale Factor:  0.1856 mm/pixel
==================================================
```

---

## 5. Dataset Summary

- **Target Object:** Rigid matte black rectangular card holder ($95.0\text{ mm} \times 62.0\text{ mm}$).
- **Reference Object:** Sony Xperia 5 II smartphone ($158.0\text{ mm} \times 68.0\text{ mm}$).
- **Total Images Captured:** 76 calibrated, undistorted images.
- **Fixed Split (Committed Prior to Training):**
  - **Train Split (70%):** 53 images (`dataset/splits/train.txt`)
  - **Val Split (20%):** 15 images (`dataset/splits/val.txt`)
  - **Test Split (10%):** 8 images (`dataset/splits/test.txt` / `test_split.txt`)

---

## 6. Empirical Verification & Assessment Results

### A. Camera Calibration (Requirement #1)
- **Images:** 25 checkerboard images ($9 \times 7$ internal corners).
- **RMS Reprojection Error:** **`0.1725 pixels`** (Target: $< 0.50$ px).
- Parameters saved in `calibration/camera_params.npz`.

### B. Segmentation Model Evaluation (Requirement #3)
- **Architecture:** `LightweightUNet` (482,737 parameters, Non-YOLO).
- **Test Set mAP@0.5:** **`1.000`**
- **Test Set Recall:** **`0.999`**
- **Test Set Mean Mask IoU:** **`0.743`**

### C. Accuracy Against Calliper Ground Truth (Requirement #5)
Evaluated on 10 held-out test instances against physical calliper ground truth ($95.00\text{ mm} \times 62.00\text{ mm}$):
- Raw benchmark log: `measurement/accuracy_benchmark.csv`

### D. Breaking Condition Analysis (Requirement #8)
- Tested camera tilt angle from $0^\circ$ to $45^\circ$.
- From $0^\circ$ to $15^\circ$, error remains within $\le 1.29\text{ mm}$.
- At $\ge 30^\circ$ perspective tilt, the system's guardrail automatically triggers **`REFUSED (REFERENCE NOT FOUND)`**, preventing false measurements.
- Log: `measurement/tilt_stress_test.csv`

---

## 8. Video Demonstration (Assessment Requirement #7 & #10)

Continuous 5-minute technical demonstration covering the physical setup, code execution, metrology verification, and oral responses to questions K1–K4:
👉 **Loom Recording:** [https://www.loom.com/share/be2d431ad7ad4673871b3818f996b151](https://www.loom.com/share/be2d431ad7ad4673871b3818f996b151)


```text
├── calibration/             # Calibration scripts, images, parameters, report
├── dataset/                 # 76 images, masks, and immutable split lists
├── models/                  # Lightweight U-Net code, weights, eval metrics
├── inference/               # Single-command measurement CLI
├── measurement/             # Calliper benchmarks, tilt stress test
├── docs/                    # Plan.md, Objectives.md, Definition_of_Done.md
├── requirements.txt         # Pinned project dependencies
└── README.md                # System documentation
```

