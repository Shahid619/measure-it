# Measurable Objectives

This document defines the quantitative targets for the **Measure It** computer vision assessment according to Section 6.

---

## Objective MO-1: Measurement Accuracy (Mandatory)

| Field | Entry |
| :--- | :--- |
| **ID** | **MO-1** |
| **What is measured** | Absolute measurement error (in millimetres) and Mean Absolute Percentage Error (MAPE) of the card holder's length and width, compared against physical calliper ground truth. |
| **How** | An automated script (`measurement/evaluate_accuracy.py`) compares the system output ($W_{\text{pred}}, H_{\text{pred}}$) against manual calliper ground truth ($W_{\text{true}} = 62.00\text{ mm}, H_{\text{true}} = 95.00\text{ mm}$) recorded across 10 held-out test instances. |
| **Target** | **Mean Absolute Error (MAE) $\le 2.00\text{ mm}$** and **Mean Absolute Percentage Error $\le 3.0\%$** across test instances. Maximum single-instance error $\le 3.50\text{ mm}$. |
| **Conditions** | Fixed-focus smartphone camera, undistorted frames, planar top-down perspective ($\le 15^\circ$ tilt from normal), co-planar Sony Xperia 5 II reference phone lying within frame, ambient indoor lighting without harsh specular glare. |
| **Not included** | Partial object occlusions, out-of-frame object boundaries, severe out-of-plane tilts ($> 25^\circ$), or cases where the reference object is missing or partially covered. |

---

## Objective MO-2: Camera Undistortion & Reprojection Fidelity

| Field | Entry |
| :--- | :--- |
| **ID** | **MO-2** |
| **What is measured** | Root Mean Square (RMS) reprojection error from OpenCV camera calibration across all detected checkerboard corners. |
| **How** | `calibration/calibrate_camera.py` computes total reprojection error by projecting 3D object points back into image space using the calculated camera matrix ($K$) and distortion coefficients ($D$), then measuring Euclidean pixel error against detected corner coordinates. |
| **Target** | **RMS Reprojection Error $< 0.50\text{ pixels}$** across $\ge 20$ calibration captures. |
| **Conditions** | Displayed checkerboard ($9 \times 7$ internal corners, square size measured with calliper), captured at varied inclinations ($0^\circ$ to $35^\circ$), distances ($25\text{ cm}$ to $70\text{ cm}$), and orientations filling the sensor frame. |
| **Not included** | Blurry frames or frames where checkerboard corners fall outside the image sensor boundaries. |

---

## Objective MO-3: Segmentation Mask Quality

| Field | Entry |
| :--- | :--- |
| **ID** | **MO-3** |
| **What is measured** | Mean Intersection over Union (mIoU), mAP@0.5, and mask precision/recall on the held-out test split. |
| **How** | Computed on the fixed test split ($10\%$ of dataset, strictly unobserved during training and validation) comparing predicted binary segmentation masks against ground-truth polygon labels. |
| **Target** | **Mask IoU $\ge 0.70$**, **mAP@0.5 $\ge 0.90$**, and **Recall $\ge 0.95$** on the test split. |
| **Conditions** | Undistorted test images at native evaluation resolution ($1280 \times 960$), identical preprocessing pipeline. |
| **Not included** | Highly degraded or corrupted images. |

---

## System Telemetry & Benchmark Environment
All metrics reported in the evaluation tables are generated directly from the execution machine:
- **Operating System:** Windows 10/11 x86_64
- **CPU:** Intel/AMD x86_64
- **RAM:** System Host Memory
- **Camera Device:** Smartphone Camera (Fixed focus, locked digital zoom 1.0x)
- **Native Resolution:** $1280 \times 960$ px
- **Target Object:** Matte black card holder ($95.0\text{ mm} \times 62.0\text{ mm}$)
- **Reference Object:** Sony Xperia 5 II ($158.0\text{ mm} \times 68.0\text{ mm}$)
