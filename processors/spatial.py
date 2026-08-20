"""Spatial-domain point operations.

Spatial-domain processing works directly on pixel values (or on a small
neighbourhood of pixels) rather than on a frequency-domain transform.
Everything in this file is a *point* operation: the output pixel depends
only on the corresponding input pixel.
"""

import cv2
import numpy as np

from .utils import ensure_gray


def to_grayscale(image):
    """RGB -> Grayscale using the luminosity weights Y = 0.299R + 0.587G + 0.114B."""
    return ensure_gray(image)


def negative(image):
    """Photographic negative:  output = 255 - input.

    Dark regions become bright and vice versa; useful for inspecting
    detail hidden in the dark parts of an image.
    """
    return 255 - np.asarray(image).astype(np.uint8)


def threshold_image(image, threshold=127, max_value=255, invert=False):
    """Global binary thresholding.

    output(x, y) = max_value  if input(x, y) > threshold
                 = 0          otherwise
    """
    gray = ensure_gray(image)
    threshold = int(np.clip(threshold, 0, 255))
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, result = cv2.threshold(gray, threshold, int(max_value), mode)
    return result


def otsu_threshold(image):
    """Otsu's method: automatically picks the threshold that minimises the
    intra-class variance of the two pixel groups.

    Returns (binary_image, chosen_threshold).
    """
    gray = ensure_gray(image)
    value, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return result, float(value)


def adaptive_threshold(image, block_size=11, constant=2):
    """Adaptive (local) thresholding - a different threshold per neighbourhood.
    Handles uneven illumination far better than a single global threshold.
    """
    gray = ensure_gray(image)
    block_size = int(block_size)
    if block_size < 3:
        block_size = 3
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, int(constant),
    )


def contrast_stretch(image, low_percentile=2.0, high_percentile=98.0):
    """Linear contrast stretching (min-max normalisation).

    output = (input - low) * 255 / (high - low)

    Percentiles are used instead of the absolute min/max so that a few
    stray outlier pixels cannot flatten the whole stretch.
    """
    array = np.asarray(image).astype(np.float32)
    low = np.percentile(array, low_percentile)
    high = np.percentile(array, high_percentile)
    if high - low < 1e-6:
        return np.asarray(image).astype(np.uint8, copy=True)
    stretched = (array - low) * (255.0 / (high - low))
    return np.clip(stretched, 0, 255).astype(np.uint8)


def histogram_equalization(image):
    """Histogram equalisation - redistributes intensities so the histogram
    becomes approximately flat, which globally improves contrast.
    """
    array = np.asarray(image)
    if array.ndim == 2:
        return cv2.equalizeHist(array)
    # For colour images equalise only the luminance channel so hues survive.
    ycrcb = cv2.cvtColor(array, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def bit_plane(image, plane=7):
    """Extract a single bit-plane. Plane 7 (MSB) holds most of the visual
    structure; plane 0 (LSB) is mostly noise.
    """
    gray = ensure_gray(image)
    plane = int(np.clip(plane, 0, 7))
    return ((gray >> plane) & 1).astype(np.uint8) * 255


def histogram_data(image, bins=256):
    """Return (counts, bin_edges) of the grayscale histogram."""
    gray = ensure_gray(image)
    counts, edges = np.histogram(gray.ravel(), bins=bins, range=(0, 256))
    return counts, edges
