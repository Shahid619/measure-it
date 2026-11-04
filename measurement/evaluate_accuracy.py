import os
import glob
import pandas as pd
import numpy as np
import cv2
from inference.measure import segment_and_measure, detect_reference_scale, load_calibration, load_segmentation_model
import torch

# Standard Ground Truth Dimensions for the Card Holder
# Measured with vernier calliper: Height = 95.0 mm, Width = 62.0 mm
TRUE_HEIGHT_MM = 95.00
TRUE_WIDTH_MM = 62.00

def evaluate_accuracy_benchmark(
    test_split_file="dataset/splits/test.txt",
    images_dir="dataset/images",
    raw_images_dir="dataset/raw_images",
    output_csv="measurement/accuracy_benchmark.csv",
    output_comparison_csv="measurement/undistortion_impact_comparison.csv"
):
    with open(test_split_file, "r") as f:
        test_files = [line.strip() for line in f if line.strip()]

    # If test split has fewer than 10, add additional held-out validation instances to meet >= 10 requirement
    all_eval_files = test_files.copy()
    if len(all_eval_files) < 10:
        with open("dataset/splits/val.txt", "r") as f:
            val_files = [line.strip() for line in f if line.strip()]
        all_eval_files.extend(val_files[:10 - len(all_eval_files)])

    print(f"Running accuracy benchmark on {len(all_eval_files)} instances against calliper ground truth...")

    records = []
    raw_vs_undist_records = []

    K, D = load_calibration()
    model = load_segmentation_model()

    for idx, filename in enumerate(all_eval_files):
        img_path = os.path.join(images_dir, filename)
        
        # 1. Undistorted Pipeline measurement
        res = segment_and_measure(img_path)
        if res["status"] != "SUCCESS":
            continue

        pred_w = res["width_mm"]
        pred_h = res["height_mm"]
        conf = res["confidence"]

        err_w = abs(pred_w - TRUE_WIDTH_MM)
        err_h = abs(pred_h - TRUE_HEIGHT_MM)
        pct_err_w = (err_w / TRUE_WIDTH_MM) * 100.0
        pct_err_h = (err_h / TRUE_HEIGHT_MM) * 100.0

        records.append({
            "instance_id": idx + 1,
            "filename": filename,
            "true_width_mm": TRUE_WIDTH_MM,
            "pred_width_mm": round(pred_w, 2),
            "error_width_mm": round(err_w, 2),
            "pct_error_width": round(pct_err_w, 2),
            "true_height_mm": TRUE_HEIGHT_MM,
            "pred_height_mm": round(pred_h, 2),
            "error_height_mm": round(err_h, 2),
            "pct_error_height": round(pct_err_h, 2),
            "mean_abs_error_mm": round((err_w + err_h) / 2.0, 2),
            "confidence": round(conf, 4)
        })

        # 2. Raw Distorted Image measurement for Section 3 Requirement 6
        # Locate corresponding raw image
        raw_list = sorted(glob.glob(os.path.join(raw_images_dir, "*.*")))
        # Frame index from frame_XXX.png
        frame_idx = int(filename.split("_")[1].split(".")[0])
        raw_path = raw_list[frame_idx]
        raw_img = cv2.imread(raw_path)

        # Scale detection on raw image (no undistortion)
        scale_raw, _, _ = detect_reference_scale(raw_img)
        if scale_raw is not None:
            # Segment on raw image
            rgb_raw = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb_raw, (256, 256))
            inp = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            with torch.no_grad():
                prob_raw = torch.sigmoid(model(inp)).squeeze().cpu().numpy()
            full_prob_raw = cv2.resize(prob_raw, (raw_img.shape[1], raw_img.shape[0]))
            mask_raw = (full_prob_raw > 0.50).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(mask_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_c = [c for c in cnts if cv2.contourArea(c) > 15000]
            if valid_c:
                target_raw = max(valid_c, key=cv2.contourArea)
                (wr, hr) = cv2.minAreaRect(target_raw)[1]
                raw_pred_w = min(wr, hr) * scale_raw
                raw_pred_h = max(wr, hr) * scale_raw
                raw_mae = (abs(raw_pred_w - TRUE_WIDTH_MM) + abs(raw_pred_h - TRUE_HEIGHT_MM)) / 2.0
                undist_mae = (err_w + err_h) / 2.0

                raw_vs_undist_records.append({
                    "instance_id": idx + 1,
                    "filename": filename,
                    "raw_distorted_mae_mm": round(raw_mae, 2),
                    "undistorted_mae_mm": round(undist_mae, 2),
                    "error_reduction_mm": round(raw_mae - undist_mae, 2),
                    "pct_improvement": round(((raw_mae - undist_mae) / max(raw_mae, 1e-3)) * 100.0, 1)
                })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nAccuracy benchmark saved to {output_csv}")

    df_comp = pd.DataFrame(raw_vs_undist_records)
    df_comp.to_csv(output_comparison_csv, index=False)
    print(f"Undistortion impact comparison saved to {output_comparison_csv}")

    print("\n" + "="*60)
    print("METROLOGY ACCURACY SUMMARY (AGAINST CALLIPER GROUND TRUTH):")
    print(f"  Instances Evaluated:       {len(df)}")
    print(f"  Mean Absolute Error (MAE): {df['mean_abs_error_mm'].mean():.2f} mm")
    print(f"  Median Absolute Error:     {df['mean_abs_error_mm'].median():.2f} mm")
    print(f"  Worst-Case Error:          {df['mean_abs_error_mm'].max():.2f} mm")
    print(f"  Mean Percentage Error:     {((df['pct_error_width'] + df['pct_error_height'])/2).mean():.2f}%")
    print("="*60)

    print("\n" + "="*60)
    print("PROOF THAT UNDISTORTION MATTERS (SIDE-BY-SIDE):")
    print(f"  Raw (Distorted) Average MAE:   {df_comp['raw_distorted_mae_mm'].mean():.2f} mm")
    print(f"  Undistorted Average MAE:       {df_comp['undistorted_mae_mm'].mean():.2f} mm")
    print(f"  Overall Error Reduction:       {df_comp['error_reduction_mm'].mean():.2f} mm ({df_comp['pct_improvement'].mean():.1f}% improvement)")
    print("="*60)

if __name__ == "__main__":
    evaluate_accuracy_benchmark()
