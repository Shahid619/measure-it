# Measure It — Project Execution Plan

## 1. Project Overview & Scope
- **Target Object:** Rigid matte black rectangular card holder with central circular aperture ($95.0\text{ mm} \times 62.0\text{ mm}$).
- **Reference Object:** Sony Xperia 5 II smartphone ($158.0\text{ mm} \times 68.0\text{ mm}$) co-planar to target.
- **Goal:** Automated optical measurement system reporting true physical width and height in millimetres with proven accuracy against calliper ground truth.

---

## 2. Time Allocation (8 Hours Working Budget)
*Note: Wall-clock training runs and dependency downloads execute in the background and are excluded from active work time.*

| Stage | Activity | Estimated Time | Priority |
| :--- | :--- | :--- | :--- |
| **Stage 1** | **Camera Calibration:** Capture 20+ checkerboard frames, calibrate camera via OpenCV, compute reprojection error ($< 0.5$ px), save intrinsic matrix $K$ and distortion vector $D$. | 1.0 hr | **Floor (Mandatory)** |
| **Stage 2** | **Dataset Collection & Freezing:** Capture 70+ images with fixed camera settings; generate undistorted images; freeze & commit `dataset/splits/test_split.txt` prior to training; annotate polygon masks. | 2.0 hrs | **Floor (Mandatory)** |
| **Stage 3** | **Segmentation Architecture & Training:** Configure PyTorch segmentation pipeline (Lightweight U-Net; non-YOLO, non-Roboflow). Train model; evaluate on held-out test split (IoU, mAP, precision, recall). | 1.5 hrs | **Floor (Mandatory)** |
| **Stage 4** | **Measurement Pipeline:** Implement single-entrypoint CLI (`inference/measure.py`). Undistorts input, infers mask, segments reference phone, calculates scale ratio ($\text{px}/\text{mm}$), extracts oriented bounding box (`cv2.minAreaRect`), outputs annotated visual + mm measurements. Implement rejection logic. | 1.5 hrs | **Floor (Mandatory)** |
| **Stage 5** | **Metrology & Empirical Validation:** Measure 10 physical instances against calliper. Compute MAE & MPE. Benchmark raw vs. undistorted accuracy. Test breaking condition (camera tilt). | 1.2 hrs | **High** |
| **Stage 6** | **Docs Verification & Loom Recording:** Finalize `README.md`, update `Definition_of_Done.md` with links/evidence, rehearse and record $\le 5$-minute Loom walk answering K1–K4. | 0.8 hr | **High** |

---

## 3. Explicit Scope Trade-offs
- **What we do:** Full 2D planar metric measurement pipeline with rigorous lens distortion removal, deep learning contour segmentation, sub-pixel corner reference detection, and error benchmarking against physical callipers.
- **What we skip:** 
  - 3D point cloud reconstruction / stereo cameras (out of scope for single monocular top-down industrial inspection).
  - Heavy multi-gigabyte models (prefer lightweight mobile backbones for reproducible CPU/GPU inference).
  - Web UI / cloud hosting (the assessment strictly requires a clean, reproducible CLI command).

---

## 4. Architectural Decision Record (ADR)

### ADR-01: Reference Scale Extraction
- **Decision:** Use a co-planar Sony Xperia 5 II smartphone with verified manufacturer dimensions ($158.0\text{ mm} \times 68.0\text{ mm}$).
- **Alternative Rejected:** Pre-calibrated fixed focal length distance equation ($Z = \frac{f \cdot X}{x}$).
- **Why Rejected:** Fixed-distance calculation assumes the camera is mounted at an exact, unwavering height on an industrial rig. A co-planar reference object allows handheld or varied-height inspection without sacrificing sub-millimeter precision.

### ADR-02: Segmentation Framework
- **Decision:** Custom PyTorch `LightweightUNet` with skip connections (482,737 parameters).
- **Alternative Rejected:** Ultralytics YOLOv8-seg / YOLOv11-seg.
- **Why Rejected:** Strictly prohibited by Section 4.2 / Section 9 automatic rejection rules. PyTorch provides direct architectural transparency, custom loss formulation, and verifiable evaluation metrics.

### ADR-03: Measurement Dimension Extraction
- **Decision:** Minimum-area oriented bounding box (`cv2.minAreaRect`) applied on the largest connected component of the predicted binary mask.
- **Alternative Rejected:** Axis-aligned bounding box (`cv2.boundingRect`).
- **Why Rejected:** Axis-aligned bounding boxes introduce severe dimensional inflation whenever the object is rotated relative to the camera frame. An oriented bounding box accurately captures the true primary and secondary orthogonal axes regardless of object rotation.

---

## 5. Decision Record & Scope Adjustments (Assessment Item 10)
During execution, the reference fiducial was updated from an unmeasured playing card to a precision CNC-manufactured Sony Xperia 5 II smartphone ($158.0\text{ mm} \times 68.0\text{ mm}$). This decision eliminated manual measurement variance of reference cards and provided rigid dimensional ground truth across all 76 captures.
