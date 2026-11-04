import os
import glob
import cv2
import pandas as pd
import numpy as np
from inference.measure import segment_and_measure, load_calibration

def evaluate_tilt_breaking_condition(
    sample_image_path="dataset/images/frame_000.png",
    output_csv="measurement/tilt_stress_test.csv"
):
    """
    Evaluates Section 3 Requirement 8: One condition that breaks the system.
    We test synthetic/physical camera tilt angle theta from 0 to 45 degrees.
    As tilt angle increases, non-coplanar perspective foreshortening occurs:
    apparent dimension = true_dimension * cos(theta).
    This causes measurement error to diverge predictably.
    """
    img = cv2.imread(sample_image_path)
    if img is None:
        raise FileNotFoundError(sample_image_path)

    h, w = img.shape[:2]
    tilt_angles_deg = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    results = []

    os.makedirs("measurement/tilt_samples", exist_ok=True)

    print("Evaluating breaking condition: Camera Tilt Angle vs. Measurement Error...")

    for angle in tilt_angles_deg:
        # Perspective tilt homography transformation
        rad = np.radians(angle)
        cos_a = np.cos(rad)
        sin_a = np.sin(rad)

        # 3D rotation around horizontal axis
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        # Compress vertical coordinate and apply trapezoidal perspective warp
        compress_factor = cos_a
        keystone = sin_a * 0.15

        dst_pts = np.float32([
            [w * keystone, 0],
            [w * (1 - keystone), 0],
            [w, h * compress_factor],
            [0, h * compress_factor]
        ])

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(img, M, (w, h), borderValue=(255, 255, 255))
        
        tilt_img_path = f"measurement/tilt_samples/tilt_{angle}deg.png"
        cv2.imwrite(tilt_img_path, warped)

        # Run measurement pipeline
        res = segment_and_measure(tilt_img_path)
        if res["status"] == "SUCCESS":
            w_err = abs(res["width_mm"] - 62.0)
            h_err = abs(res["height_mm"] - 95.0)
            mae = (w_err + h_err) / 2.0
            results.append({
                "tilt_angle_deg": angle,
                "status": "ACCEPTED",
                "width_mm": round(res["width_mm"], 2),
                "height_mm": round(res["height_mm"], 2),
                "mae_error_mm": round(mae, 2),
                "pct_error": round((mae / 78.5) * 100.0, 2),
                "confidence": round(res["confidence"], 4)
            })
        else:
            results.append({
                "tilt_angle_deg": angle,
                "status": f"REFUSED ({res['error']})",
                "width_mm": None,
                "height_mm": None,
                "mae_error_mm": None,
                "pct_error": None,
                "confidence": 0.0
            })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Tilt breaking analysis saved to: {output_csv}")
    print("\nTILT STRESS TEST RESULTS:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    evaluate_tilt_breaking_condition()
