# Definition of Done

This checklist defines the empirical criteria for project completion. Every checked box must link directly to an artifact, commit, or terminal log output. Unchecked boxes or checked boxes lacking evidence count as incomplete.

---

## Verification Checklist

- [ ] **1. Camera Calibration Reprojection Error**
  - *Target:* RMS Reprojection Error $< 0.50$ px across $\ge 20$ calibration captures.
  - *Evidence:* Reported in `calibration/calibration_report.json` and saved matrices in `calibration/camera_params.npz`.
  - *Result:* `Pending calibration run`

- [ ] **2. Test Split Committed Prior to Model Training**
  - *Target:* Split file list `dataset/splits/test.txt` committed to Git history strictly before any training scripts or weights.
  - *Evidence:* Git commit SHA: `Pending commit`

- [ ] **3. Segmentation Model Trained (Non-YOLO / Non-Roboflow)**
  - *Target:* Architecture complies with Section 4.2. Evaluated on test split: Mask IoU, precision, recall, and loss curve.
  - *Evidence:* `models/eval_metrics.json` and saved model checkpoint.
  - *Result:* `Pending training run`

- [ ] **4. Single-Command Inference (`image-to-millimetres`)**
  - *Target:* Command takes a single image path, applies undistortion, segments card holder, finds reference card, extracts scale, and prints width (mm), height (mm), and confidence score. Handles missing object and missing reference cleanly.
  - *Evidence:* CLI invocation log and output overlay image.
  - *Command:* `python -m inference.measure --image <path>`

- [ ] **5. Calliper Ground Truth Accuracy Table (10+ Instances)**
  - *Target:* Per-instance table comparing physical measurement vs. model prediction. Report Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE).
  - *Evidence:* `measurement/accuracy_table.csv` and summary in `README.md`.
  - *Result:* `Pending evaluation`

- [ ] **6. Proof of Lens Undistortion Impact**
  - *Target:* Side-by-side comparison measuring identical 10 instances on raw distorted vs. undistorted images. Document error divergence with empirical data.
  - *Evidence:* Comparative metrics table in `README.md`.
  - *Result:* `Pending evaluation`

- [ ] **7. Failure Mode & Boundary Analysis**
  - *Target:* Controlled stress test of one degrading environmental factor (camera tilt angle from $0^\circ$ to $45^\circ$) showing error divergence.
  - *Evidence:* `measurement/tilt_stress_test.csv` and degradation curve plot.
  - *Result:* `Pending evaluation`

- [ ] **8. System Rejection & Trust Guardrails**
  - *Target:* Explicit refusal to predict dimensions when reference object is absent/occluded or mask confidence falls below confidence threshold $\tau$.
  - *Evidence:* Rejection test unit and sample console logs.
  - *Result:* `Pending evaluation`

- [ ] **9. Code Quality & Git Hygiene**
  - *Target:* Progressive commit history on branch `dev-test01` without bulk commits or wip messages. Clear repository structure.
  - *Evidence:* Git log inspection.

- [ ] **10. Final Loom Demonstration (≤ 5 minutes)**
  - *Target:* Continuous screen recording including live measurement against calliper, code walkthrough, and oral response to technical questions K1–K4.
  - *Evidence:* Loom URL submitted in PR description.
  - *Result:* `Pending recording`

---

## Out-of-Scope & Incomplete Declarations
*(Any item not reached by the 8-hour cutoff will be documented here with justification.)*
- None declared at project start.
