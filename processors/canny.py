"""Canny edge detection.

Canny is a multi-stage optimal edge detector:

    Input image
        -> Gaussian smoothing          (remove noise)
        -> Gradient calculation        (Sobel Gx, Gy)
        -> Magnitude & direction       (sqrt(Gx^2+Gy^2), arctan2(Gy,Gx))
        -> Non-maximum suppression     (thin the ridges to 1 pixel)
        -> Double thresholding         (strong / weak / suppressed)
        -> Hysteresis edge tracking    (keep weak edges touching strong ones)
        -> Final binary edge image

OpenCV's cv2.Canny performs stages 2-6 internally; the Gaussian
smoothing stage is applied here so it stays adjustable.
"""

import cv2
import numpy as np

from . import gradients
from .utils import ensure_gray, normalize_to_uint8, odd_kernel

# Presets offered in the UI (low, high). A 1:2 - 1:3 ratio between the two
# thresholds gives the most stable results.
PRESETS = {
    "Fine Edges": (50, 100),
    "Balanced": (100, 200),
    "Strong Edges": (150, 250),
}


def canny_edge_detection(image, low_threshold=100, high_threshold=200,
                         blur_kernel=5, aperture_size=3, l2_gradient=True):
    """Full Canny pipeline.

    blur_kernel  : Gaussian pre-smoothing kernel (use 1 to skip smoothing)
    aperture_size: Sobel aperture used inside Canny (3, 5 or 7)
    l2_gradient  : True  -> exact sqrt(Gx^2 + Gy^2)
                   False -> faster approximation |Gx| + |Gy|
    """
    gray = ensure_gray(image)

    # Stage 1 - Gaussian smoothing.
    blur_kernel = odd_kernel(blur_kernel, minimum=1)
    if blur_kernel > 1:
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    low = int(np.clip(low_threshold, 0, 255))
    high = int(np.clip(high_threshold, 0, 255))
    if low > high:  # keep OpenCV happy even if the UI warning is ignored
        low, high = high, low

    aperture_size = int(aperture_size)
    if aperture_size not in (3, 5, 7):
        aperture_size = 3

    # Stages 2-6 are performed by OpenCV.
    return cv2.Canny(gray, low, high, apertureSize=aperture_size,
                     L2gradient=bool(l2_gradient))


def canny_stages(image, low_threshold=100, high_threshold=200,
                 blur_kernel=5, aperture_size=3):
    """Return the intermediate images so each stage can be shown separately.

    Non-maximum suppression and hysteresis happen inside cv2.Canny, so the
    stages exposed here are: smoothed input, gradient magnitude, gradient
    direction visualisation, the strong-only edge map (both thresholds set
    to the high value, i.e. no hysteresis) and the final result.
    """
    gray = ensure_gray(image)
    blur_kernel = odd_kernel(blur_kernel, minimum=1)
    smoothed = (cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
                if blur_kernel > 1 else gray.copy())

    grad = gradients.sobel_operator(smoothed, min(aperture_size, 7))
    direction_rgb = gradients.direction_visualization(
        grad["magnitude_raw"], grad["direction_raw"]
    )

    high = int(np.clip(high_threshold, 0, 255))
    strong_only = cv2.Canny(smoothed, high, high, apertureSize=int(aperture_size))
    final = canny_edge_detection(image, low_threshold, high_threshold,
                                 blur_kernel, aperture_size)

    return {
        "Gaussian Smoothed": smoothed,
        "Gradient Magnitude": normalize_to_uint8(grad["magnitude_raw"]),
        "Gradient Direction": direction_rgb,
        "Strong Edges Only (no hysteresis)": strong_only,
        "Final Canny Result": final,
    }


def compare_presets(image, blur_kernel=5, aperture_size=3):
    """Run every preset so the effect of the thresholds can be compared."""
    results = {}
    for name, (low, high) in PRESETS.items():
        label = "%s (%d / %d)" % (name, low, high)
        results[label] = canny_edge_detection(
            image, low, high, blur_kernel, aperture_size
        )
    return results


def edge_pixel_ratio(edge_image):
    """Percentage of pixels marked as edges - a quick quantitative measure
    of how aggressive the current thresholds are.
    """
    edges = np.asarray(edge_image)
    if edges.size == 0:
        return 0.0
    return float(np.count_nonzero(edges) / edges.size * 100.0)
