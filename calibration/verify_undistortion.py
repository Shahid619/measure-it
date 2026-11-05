import os
import cv2
import numpy as np

def verify_undistortion(
    sample_image_path="calibration/images/WhatsApp Image 2026-11-04 at 6.46.22 PM.jpeg",
    params_path="calibration/camera_params.npz",
    output_comparison_path="calibration/undistortion_comparison.png"
):
    """
    Loads saved camera parameters, undistorts a sample calibration image,
    draws reference grid lines, and saves side-by-side verification.
    """
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Parameters not found: {params_path}")

    data = np.load(params_path)
    K = data["camera_matrix"]
    D = data["dist_coeffs"]

    img = cv2.imread(sample_image_path)
    if img is None:
        raise FileNotFoundError(f"Sample image not found: {sample_image_path}")

    h, w = img.shape[:2]
    # Optimal new camera matrix
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    undistorted = cv2.undistort(img, K, D, None, new_K)

    # Draw reference straight lines across both to visually inspect straightness
    raw_drawn = img.copy()
    undist_drawn = undistorted.copy()

    # Draw horizontal grid lines across the image
    for y in range(h // 6, h, h // 6):
        cv2.line(raw_drawn, (0, y), (w, y), (0, 0, 255), 2)
        cv2.line(undist_drawn, (0, y), (w, y), (0, 255, 0), 2)

    cv2.putText(raw_drawn, "RAW (Distorted)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.putText(undist_drawn, "UNDISTORTED (Corrected)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    combined = np.hstack([raw_drawn, undist_drawn])
    cv2.imwrite(output_comparison_path, combined)
    print(f"Undistortion verification saved to: {output_comparison_path}")

if __name__ == "__main__":
    verify_undistortion()

