"""
Simple Error Level Analysis (ELA)
Image Tampering Detection Hackathon

How it works:
1. Load the original image.
2. Recompress it as JPEG.
3. Compare the original and recompressed images.
4. Calculate pixel differences.
5. Amplify the differences.
6. Generate an ELA heatmap.
7. Calculate an overall ELA score.
"""

from PIL import Image
import numpy as np
from io import BytesIO
import os
import sys


def perform_ela(image, quality=90):
    """
    Perform Error Level Analysis.

    Parameters:
        image: image file path or PIL Image
        quality: JPEG compression quality

    Returns:
        ela_heatmap: ELA heatmap image
        ela_score: overall ELA score
    """

    # --------------------------------------------------
    # STEP 1: Load the original image
    # --------------------------------------------------

    if isinstance(image, str):
        original = Image.open(image).convert("RGB")
    else:
        original = image.convert("RGB")

    # --------------------------------------------------
    # STEP 2: Recompress the image as JPEG
    # --------------------------------------------------

    buffer = BytesIO()

    original.save(
        buffer,
        format="JPEG",
        quality=quality
    )

    buffer.seek(0)

    recompressed = Image.open(buffer).convert("RGB")

    # --------------------------------------------------
    # STEP 3: Convert images to NumPy arrays
    # --------------------------------------------------

    original_array = np.asarray(original).astype(np.int16)

    recompressed_array = np.asarray(recompressed).astype(np.int16)

    # --------------------------------------------------
    # STEP 4: Calculate pixel differences
    # --------------------------------------------------

    difference = np.abs(
        original_array - recompressed_array
    )

    # Take the largest difference among R, G and B
    difference = np.max(difference, axis=2)

    # --------------------------------------------------
    # STEP 5: Calculate ELA score
    # --------------------------------------------------

    ela_score = float(
        np.mean(difference) / 255.0 * 100.0
    )

    # --------------------------------------------------
    # STEP 6: Amplify differences
    # --------------------------------------------------

    max_difference = difference.max()

    if max_difference > 0:

        scale = 255.0 / max_difference

        ela_array = np.clip(
            difference * scale,
            0,
            255
        ).astype(np.uint8)

    else:

        ela_array = np.zeros_like(
            difference,
            dtype=np.uint8
        )

    # --------------------------------------------------
    # STEP 7: Create ELA heatmap
    # --------------------------------------------------

    gray = ela_array.astype(np.float32) / 255.0

    heatmap = np.zeros(
        (gray.shape[0], gray.shape[1], 3),
        dtype=np.uint8
    )

    # Red channel
    heatmap[:, :, 0] = np.clip(
        gray * 255,
        0,
        255
    )

    # Green channel
    heatmap[:, :, 1] = np.clip(
        (gray - 0.35) / 0.65 * 255,
        0,
        255
    )

    # Blue channel
    heatmap[:, :, 2] = np.clip(
        (gray - 0.75) / 0.25 * 255,
        0,
        255
    )

    ela_heatmap = Image.fromarray(
        heatmap,
        mode="RGB"
    )

    # Return heatmap and score
    return ela_heatmap, ela_score


# ======================================================
# SAVE ELA RESULTS
# ======================================================

def save_ela(
    image_path,
    output_path="ela_output.png",
    overlay_path="ela_overlay.png"
):

    # Run ELA
    heatmap, score = perform_ela(image_path)

    # Save heatmap
    heatmap.save(output_path)

    # Load original image
    original = Image.open(image_path).convert("RGB")

    # Create overlay
    overlay = Image.blend(
        original,
        heatmap,
        alpha=0.55
    )

    # Save overlay
    overlay.save(overlay_path)

    # Display results
    print("--------------------------------")
    print("ELA ANALYSIS COMPLETE")
    print("--------------------------------")

    print(f"ELA Score: {score:.2f}")

    print(f"Heatmap saved to: {output_path}")

    print(f"Overlay saved to: {overlay_path}")

    print("--------------------------------")


# ======================================================
# MAIN PROGRAM
# ======================================================

if __name__ == "__main__":

    # Check whether image path was provided

    if len(sys.argv) < 2:

        print("Please provide an image.")

        print("Example:")

        print("python ela.py image.jpg")

        sys.exit(1)

    image_path = sys.argv[1]

    # Check whether image exists

    if not os.path.exists(image_path):

        print(f"ERROR: Image not found: {image_path}")

        sys.exit(1)

    # Run ELA

    save_ela(image_path)
