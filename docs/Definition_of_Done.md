# Definition of Done

This checklist defines the empirical criteria for project completion. Every checked box must link directly to an artifact, commit, or terminal log output. Unchecked boxes or checked boxes lacking evidence count as incomplete.

---

## Verification Checklist

- [x] **1. Camera Calibration Reprojection Error**
  - *Target:* RMS Reprojection Error $< 0.50$ px across $\ge 20$ calibration captures.
  - *Evidence:* Reported in `calibration/calibration_report.json` and saved matrices in `calibration/camera_params.npz`.
  - *Result:* **0.1725 px** RMS error across 25/25 valid calibration images (Target met). Visual verification: `calibration/undistortion_comparison.png`.

- [x] **2. Test Split Committed Prior to Model Training**
  - *Target:* Split file list `dataset/splits/test.txt` committed to Git history strictly before any training scripts or weights.
  - *Evidence:* Git commit SHA `c07c949` committed before model training commit `9d27a67`.

- [x] **3. Segmentation Model Trained (Non-YOLO / Non-Roboflow)**
  - *Target:* Architecture complies with Section 4.2. Evaluated on test split: Mask IoU, precision, recall, and loss curve.
  - *Evidence:* `models/eval_metrics.json` and checkpoint `models/cardholder_unet.pth`.
  - *Result:* Test mAP@0.5 = 1.0, Recall = 0.999, Mask IoU = 0.743.

- [x] **4. Single-Command Inference (`image-to-millimetres`)**
  - *Target:* Command takes a single image path, applies undistortion, segments card holder, finds reference card, extracts scale, and prints width (mm), height (mm), and confidence score. Handles missing object and missing reference cleanly.
  - *Evidence:* `python -m inference.measure --image dataset/images/frame_000.png`. Output visual: `inference/output_measurement.png`. Both guardrails verified.

- [x] **5. Calliper Ground Truth Accuracy Table (10+ Instances)**
  - *Target:* Per-instance table comparing physical measurement vs. model prediction. Report Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE).
  - *Evidence:* `measurement/accuracy_benchmark.csv`. Evaluated across 10 instances.

- [x] **6. Proof of Lens Undistortion Impact**
  - *Target:* Side-by-side comparison measuring identical 10 instances on raw distorted vs. undistorted images. Document error divergence with empirical data.
  - *Evidence:* `measurement/undistortion_impact_comparison.csv`.

- [x] **7. Failure Mode & Boundary Analysis**
  - *Target:* Controlled stress test of one degrading environmental factor (camera tilt angle from $0^\circ$ to $45^\circ$) showing error divergence.
  - *Evidence:* `measurement/tilt_stress_test.csv` demonstrating reliable measurement below $25^\circ$ and automatic guardrail refusal at $\ge 30^\circ$.

- [x] **8. System Rejection & Trust Guardrails**
  - *Target:* Explicit refusal to predict dimensions when reference object is absent/occluded or mask confidence falls below confidence threshold $\tau$.
  - *Evidence:* Automated tests in `inference/measure.py` returning `REFUSED (REFERENCE NOT FOUND)` and `REFUSED (OBJECT NOT FOUND)`.

- [x] **9. Code Quality & Git Hygiene**
  - *Target:* Progressive commit history on branch `dev-test01` without bulk commits or wip messages. Clear repository structure.
  - *Evidence:* Clean git log on `dev-test01` with ordered, semantic commits.

- [x] **10. Final Loom Demonstration (≤ 5 minutes)**
  - *Target:* Continuous screen recording including live measurement against calliper, code walkthrough, and oral response to technical questions K1–K4.
  - *Evidence:* Loom URL: https://www.loom.com/share/be2d431ad7ad4673871b3818f996b151
  - *Result:* Completed and verified in under 5 minutes.

---

## Out-of-Scope & Incomplete Declarations
*(Any item not reached by the 8-hour cutoff will be documented here with justification.)*
- None declared at project start.

