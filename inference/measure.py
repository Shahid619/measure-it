import os
import argparse
import json
import cv2
import numpy as np
import torch
from models.unet import LightweightUNet

# Reference object physical dimensions (Sony Xperia 5 II)
REF_PHONE_HEIGHT_MM = 158.0
REF_PHONE_WIDTH_MM = 68.0

def load_calibration(params_path="calibration/camera_params.npz"):
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Calibration file missing: {params_path}")
    data = np.load(params_path)
    return data["camera_matrix"], data["dist_coeffs"]

def load_segmentation_model(weights_path="models/cardholder_unet.pth", device="cpu"):
    model = LightweightUNet()
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def detect_reference_scale(undistorted_bgr):
    """
    Detects the Sony Xperia 5 II reference phone in the undistorted image.
    Returns:
      scale_ratio (float): millimetres per pixel (e.g. ~0.18 mm/px)
      ref_box (ndarray): 4 corner points of reference bounding box
      confidence (float): detection confidence score
    """
    gray = cv2.cvtColor(undistorted_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 110, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 40000: # Phone is large (~250k - 350k px)
            rect = cv2.minAreaRect(c)
            (w, h) = rect[1]
            dim1, dim2 = max(w, h), min(w, h)
            aspect = dim1 / max(dim2, 1)
            # Sony Xperia 5 II aspect ratio = 158 / 68 = 2.32
            aspect_err = abs(aspect - 2.32)
            candidates.append((aspect_err, area, rect, c))

    if not candidates:
        return None, None, 0.0

    # Pick candidate closest to expected phone aspect ratio 2.32
    candidates.sort(key=lambda x: x[0])
    best_aspect_err, best_area, best_rect, best_contour = candidates[0]

    # Guardrail: If aspect ratio is completely broken (> 0.6 deviation), reject
    if best_aspect_err > 0.6:
        return None, None, 0.0

    (w, h) = best_rect[1]
    pixel_length = max(w, h)
    pixel_width = min(w, h)

    # Scale ratio (mm per pixel)
    scale_from_length = REF_PHONE_HEIGHT_MM / pixel_length
    scale_from_width = REF_PHONE_WIDTH_MM / pixel_width
    scale_ratio = (scale_from_length + scale_from_width) / 2.0

    ref_box = cv2.boxPoints(best_rect)
    ref_box = np.intp(ref_box)
    confidence = max(0.0, min(1.0, 1.0 - (best_aspect_err / 0.6)))

    return scale_ratio, ref_box, confidence

def segment_and_measure(
    image_path,
    output_vis_path=None,
    weights_path="models/cardholder_unet.pth",
    params_path="calibration/camera_params.npz",
    min_confidence=0.60
):
    """
    Main image-to-millimetres pipeline:
    1. Undistorts input image with OpenCV K and D
    2. Detects reference object and calculates mm/px scale
    3. Segments card holder via trained U-Net model
    4. Computes Width (mm), Height (mm), and Confidence
    5. Cleanly handles two error cases:
       - 'OBJECT NOT FOUND'
       - 'REFERENCE NOT FOUND'
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    raw_img = cv2.imread(image_path)
    if raw_img is None:
        raise ValueError(f"Cannot decode image: {image_path}")

    # 1. Undistortion
    K, D = load_calibration(params_path)
    h_orig, w_orig = raw_img.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w_orig, h_orig), 0, (w_orig, h_orig))
    undistorted = cv2.undistort(raw_img, K, D, None, new_K)

    # 2. Reference scale detection
    scale_ratio, ref_box, ref_conf = detect_reference_scale(undistorted)
    if scale_ratio is None or ref_conf < min_confidence:
        print("[GUARDRAIL REJECTION] ERROR: REFERENCE NOT FOUND")
        print("  Reason: Sony Xperia 5 II reference object could not be reliably located or scale is ambiguous.")
        return {
            "status": "REJECTED",
            "error": "REFERENCE NOT FOUND",
            "width_mm": None,
            "height_mm": None,
            "confidence": 0.0
        }

    # 3. Object Segmentation via U-Net
    model = load_segmentation_model(weights_path)
    rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (256, 256))
    inp_tensor = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

    with torch.no_grad():
        logits = model(inp_tensor)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Upsample mask to native image resolution
    full_prob = cv2.resize(prob, (w_orig, h_orig))
    binary_mask = (full_prob > 0.50).astype(np.uint8) * 255

    # 4. Extract Card Holder Dimensions (EXCLUDE any mask that overlaps with reference phone)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 15000:
            # Check overlap with reference box
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # If center is inside reference phone box, it is phone, not card holder!
                if cv2.pointPolygonTest(ref_box, (cx, cy), False) >= 0:
                    continue
            valid_contours.append(c)

    if not valid_contours:
        print("[GUARDRAIL REJECTION] ERROR: OBJECT NOT FOUND")
        print("  Reason: Card holder segmentation mask was empty or fell below minimum area threshold.")
        return {
            "status": "REJECTED",
            "error": "OBJECT NOT FOUND",
            "width_mm": None,
            "height_mm": None,
            "confidence": 0.0
        }

    # Largest predicted contour for card holder
    target_contour = max(valid_contours, key=cv2.contourArea)
    target_rect = cv2.minAreaRect(target_contour)
    (w_px, h_px) = target_rect[1]

    # Convert pixels to millimetres via reference scale ratio (LINE 145)
    dim_major_px = max(w_px, h_px)
    dim_minor_px = min(w_px, h_px)

    length_mm = dim_major_px * scale_ratio
    width_mm = dim_minor_px * scale_ratio
    overall_confidence = float(ref_conf * np.mean(full_prob[binary_mask == 255]))

    # 5. Visual output overlay
    vis = undistorted.copy()
    # Draw Reference Phone in Blue
    cv2.drawContours(vis, [ref_box], 0, (255, 0, 0), 3)
    cv2.putText(vis, "REF: Sony Xperia 5 II", (ref_box[0][0], ref_box[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Draw Card Holder in Green
    target_box = np.intp(cv2.boxPoints(target_rect))
    cv2.drawContours(vis, [target_box], 0, (0, 255, 0), 3)

    # Semi-transparent mask overlay
    mask_colored = np.zeros_like(vis)
    mask_colored[binary_mask == 255] = [0, 255, 0]
    vis = cv2.addWeighted(vis, 0.85, mask_colored, 0.15, 0)

    # Text overlay
    label_txt = f"W: {width_mm:.1f} mm | H: {length_mm:.1f} mm | Conf: {overall_confidence:.2f}"
    cv2.putText(vis, label_txt, (target_box[0][0], max(30, target_box[0][1] - 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)

    if output_vis_path:
        os.makedirs(os.path.dirname(output_vis_path), exist_ok=True)
        cv2.imwrite(output_vis_path, vis)
        print(f"Visual overlay output saved to: {output_vis_path}")

    print("\n" + "="*50)
    print("MEASUREMENT PIPELINE RESULT:")
    print(f"  Status: SUCCESS")
    print(f"  Target Object: Card Holder")
    print(f"  Width (mm):    {width_mm:.2f} mm")
    print(f"  Height (mm):   {length_mm:.2f} mm")
    print(f"  Confidence:    {overall_confidence:.4f}")
    print(f"  Scale Factor:  {scale_ratio:.4f} mm/pixel")
    print("="*50)

    return {
        "status": "SUCCESS",
        "width_mm": float(width_mm),
        "height_mm": float(length_mm),
        "confidence": float(overall_confidence),
        "scale_ratio_mm_per_px": float(scale_ratio)
    }

def main():
    parser = argparse.ArgumentParser(description="One-command image to millimetres inference pipeline.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file.")
    parser.add_argument("--output", type=str, default="inference/output_measurement.png", help="Path to save output visual overlay.")
    args = parser.parse_args()

    segment_and_measure(args.image, args.output)

if __name__ == "__main__":
    main()
