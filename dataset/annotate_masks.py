import os
import glob
import cv2
import numpy as np

def generate_segmentation_masks(
    images_dir="dataset/images",
    masks_dir="dataset/masks",
    overlay_dir="dataset/masks_preview"
):
    """
    Extracts precise ground-truth segmentation masks for the card holder.
    Identifies both objects (Phone reference and Card Holder) via high-contrast
    photometric thresholding and distinguishes the card holder by its geometry
    (smaller area ~150k px vs ~300k px for phone, aspect ratio ~1.5 vs ~2.3).
    """
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    img_files = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    print(f"Generating segmentation masks for {len(img_files)} images...")

    success = 0
    for path in img_files:
        name = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # White background (> 120), dark objects (< 100)
        # Using Otsu or adaptive/fixed threshold
        _, thresh = cv2.threshold(blurred, 110, 255, cv2.THRESH_BINARY_INV)

        # Clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 10000:
                x, y, w, h = cv2.boundingRect(c)
                aspect = max(w, h) / max(min(w, h), 1)
                candidates.append((area, aspect, c))

        # Sort by area descending
        candidates.sort(key=lambda x: x[0], reverse=True)

        # The card holder is the second largest dark component, or the one with aspect ratio ~1.4 - 1.7
        card_holder_contour = None
        if len(candidates) >= 2:
            # First is phone (~300k px), second is card holder (~150k px)
            # Verify aspect ratios
            if candidates[0][1] > candidates[1][1]:
                # Candidate 0 is phone (aspect ~2.3), Candidate 1 is card holder (aspect ~1.5)
                card_holder_contour = candidates[1][2]
            else:
                card_holder_contour = candidates[0][2]
        elif len(candidates) == 1:
            card_holder_contour = candidates[0][2]

        mask = np.zeros(gray.shape, dtype=np.uint8)
        if card_holder_contour is not None:
            cv2.drawContours(mask, [card_holder_contour], -1, 255, -1)
            # Fill any inner hole so mask represents full solid footprint
            cv2.fillPoly(mask, [card_holder_contour], 255)
            success += 1

        mask_out = os.path.join(masks_dir, name)
        cv2.imwrite(mask_out, mask)

        # Create visual overlay preview
        overlay = img.copy()
        overlay[mask == 255] = overlay[mask == 255] * 0.5 + np.array([0, 255, 0], dtype=np.uint8) * 0.5
        cv2.imwrite(os.path.join(overlay_dir, name), overlay)

    print(f"Generated masks: {success} / {len(img_files)} images processed successfully!")
    print(f"Masks saved to {masks_dir}/")
    print(f"Previews saved to {overlay_dir}/")

if __name__ == "__main__":
    generate_segmentation_masks()
