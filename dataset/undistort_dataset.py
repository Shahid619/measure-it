import os
import glob
import cv2
import numpy as np

def undistort_dataset(
    raw_dir="dataset/raw_images",
    output_dir="dataset/images",
    params_path="calibration/camera_params.npz"
):
    """
    Applies calibrated camera intrinsics to undistort all raw images.
    Saves clean, standardized images with straight planar geometry.
    """
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Parameters not found: {params_path}")

    params = np.load(params_path)
    K = params["camera_matrix"]
    D = params["dist_coeffs"]

    os.makedirs(output_dir, exist_ok=True)
    raw_files = sorted(
        glob.glob(os.path.join(raw_dir, "*.jpeg")) +
        glob.glob(os.path.join(raw_dir, "*.jpg")) +
        glob.glob(os.path.join(raw_dir, "*.png"))
    )

    print(f"Undistorting {len(raw_files)} dataset images using calibrated K and D...")

    new_K = None
    for idx, path in enumerate(raw_files):
        img = cv2.imread(path)
        if img is None:
            continue

        h, w = img.shape[:2]
        if new_K is None:
            new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0, (w, h))

        undistorted = cv2.undistort(img, K, D, None, new_K)
        out_name = f"frame_{idx:03d}.png"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, undistorted)

    print(f"Successfully saved {len(raw_files)} undistorted images to {output_dir}/")

if __name__ == "__main__":
    undistort_dataset()
