import os
import glob
import json
import cv2
import numpy as np

def calibrate_camera(
    images_dir="calibration/images",
    pattern_size=(9, 7),
    square_size_mm=25.0, # Approximate real square size for metric scale in 3D points
    output_params_file="calibration/camera_params.npz",
    output_report_file="calibration/calibration_report.json"
):
    """
    Calibrates camera intrinsics using OpenCV chessboard corner detection.
    Computes camera matrix K, distortion coefficients D, and RMS reprojection error.
    """
    # Prepare 3D object points in real-world coordinate space (Z=0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size_mm

    objpoints = [] # 3D points in real-world space
    imgpoints = [] # 2D points in image plane
    valid_images = []

    # Termination criteria for sub-pixel corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    image_paths = sorted(
        glob.glob(os.path.join(images_dir, "*.jpeg")) +
        glob.glob(os.path.join(images_dir, "*.jpg")) +
        glob.glob(os.path.join(images_dir, "*.png"))
    )

    if not image_paths:
        raise ValueError(f"No calibration images found in {images_dir}")

    image_shape = None

    print(f"Processing {len(image_paths)} calibration images...")

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_shape is None:
            image_shape = gray.shape[::-1] # (width, height)

        # Standard pattern check (9, 7)
        ret, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        # If rotated, check transposed pattern (7, 9)
        if not ret:
            ret_t, corners_t = cv2.findChessboardCorners(
                gray, (pattern_size[1], pattern_size[0]),
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            if ret_t:
                ret = True
                corners = corners_t

        if ret:
            corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners_subpix)
            valid_images.append(path)
            print(f"  [SUCCESS] {os.path.basename(path)}")
        else:
            print(f"  [REJECTED] {os.path.basename(path)}")

    print(f"\nUsable images for calibration: {len(valid_images)} / {len(image_paths)}")

    # Run OpenCV camera calibration
    ret_rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_shape, None, None
    )

    # Compute per-image and mean reprojection error
    per_image_errors = []
    total_error = 0
    total_points = 0

    for i in range(len(objpoints)):
        projected_imgpoints, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        projected_pts = projected_imgpoints.reshape(-1, 2).astype(np.float32)
        orig_pts = imgpoints[i].reshape(-1, 2).astype(np.float32)
        error = cv2.norm(orig_pts, projected_pts, cv2.NORM_L2)
        mean_img_err = error / np.sqrt(len(projected_pts))
        per_image_errors.append({
            "image": os.path.basename(valid_images[i]),
            "reprojection_error_px": float(mean_img_err)
        })
        total_error += error ** 2
        total_points += len(projected_pts)

    mean_reprojection_error = np.sqrt(total_error / total_points)

    print("\n" + "="*50)
    print("CALIBRATION RESULTS:")
    print(f"  Total valid views: {len(valid_images)}")
    print(f"  Image resolution: {image_shape[0]} x {image_shape[1]} px")
    print(f"  RMS Reprojection Error: {ret_rms:.4f} pixels")
    print(f"  Mean Euclidean Point Error: {mean_reprojection_error:.4f} pixels")
    print("="*50)

    # Save parameters for pipeline consumption
    os.makedirs(os.path.dirname(output_params_file), exist_ok=True)
    np.savez(
        output_params_file,
        camera_matrix=mtx,
        dist_coeffs=dist,
        image_shape=np.array(image_shape),
        reprojection_error=np.array([ret_rms])
    )
    print(f"Saved camera parameters to: {output_params_file}")

    # Save detailed JSON report
    report = {
        "calibrated_image_count": len(valid_images),
        "total_images_submitted": len(image_paths),
        "image_resolution": {
            "width": image_shape[0],
            "height": image_shape[1]
        },
        "rms_reprojection_error_px": float(ret_rms),
        "mean_point_error_px": float(mean_reprojection_error),
        "camera_matrix_K": mtx.tolist(),
        "distortion_coefficients_D": dist.ravel().tolist(),
        "per_image_errors": per_image_errors
    }

    with open(output_report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved calibration report to: {output_report_file}")

    return ret_rms, mtx, dist

if __name__ == "__main__":
    calibrate_camera()
