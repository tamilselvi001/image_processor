"""Small shared helpers used by every processing module.

Kept deliberately tiny: colour-space coercion, safe normalisation and
kernel validation.  Every public processing function in this package
returns a NEW array, so the uploaded image is never modified in place.
"""

import cv2
import numpy as np


def ensure_gray(image):
    """Return a single-channel uint8 copy of `image`.

    Accepts 2-D (already grayscale) or 3-D RGB / RGBA arrays.
    """
    image = np.asarray(image)
    if image.ndim == 2:
        return image.astype(np.uint8, copy=True)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def ensure_color(image):
    """Return a 3-channel RGB uint8 copy of `image`."""
    image = np.asarray(image)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image.astype(np.uint8, copy=True)


def normalize_to_uint8(array):
    """Scale any float/signed result back to a displayable 0-255 uint8 image.

    Gradient operators produce negative and out-of-range values, so they
    must be rescaled before they can be shown on screen.
    """
    array = np.asarray(array, dtype=np.float64)
    lo, hi = float(array.min()), float(array.max())
    if hi - lo < 1e-9:  # flat image -> avoid divide by zero
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = (array - lo) * (255.0 / (hi - lo))
    return scaled.astype(np.uint8)


def odd_kernel(size, minimum=1):
    """Force `size` to be an odd integer >= minimum (OpenCV requirement)."""
    size = int(size)
    if size < minimum:
        size = minimum
    if size % 2 == 0:
        size += 1
    return size


def image_info(image):
    """Return a small dict describing the image, used by the UI header cards."""
    image = np.asarray(image)
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    kind = "Grayscale" if channels == 1 else ("RGB Colour" if channels == 3 else "RGBA")
    return {
        "width": int(width),
        "height": int(height),
        "channels": int(channels),
        "dtype": str(image.dtype),
        "type": kind,
        "pixels": int(width * height),
    }


def resize_for_processing(image, max_dim=1200):
    """Downscale large images (aspect ratio preserved) to keep the app snappy.

    Returns (image, was_resized).
    """
    image = np.asarray(image)
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dim:
        return image, False
    scale = max_dim / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA), True
