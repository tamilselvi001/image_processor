"""Gradient operators: Roberts, Prewitt and Sobel.

The image gradient measures how fast intensity changes.  Each operator
approximates the partial derivatives dI/dx and dI/dy with a small
convolution mask; edges are where the gradient magnitude is large.

    magnitude = sqrt(Gx^2 + Gy^2)
    direction = arctan2(Gy, Gx)
"""

import cv2
import numpy as np

from .utils import ensure_gray, normalize_to_uint8

# --- Kernels ----------------------------------------------------------------

# Roberts cross-gradient: 2x2 diagonal differences. Cheapest and fastest,
# but very sensitive to noise because the neighbourhood is tiny.
ROBERTS_X = np.array([[1, 0],
                      [0, -1]], dtype=np.float32)
ROBERTS_Y = np.array([[0, 1],
                      [-1, 0]], dtype=np.float32)

# Prewitt: 3x3 difference with uniform averaging along the perpendicular
# direction -> some noise immunity, all neighbours weighted equally.
PREWITT_X = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]], dtype=np.float32)
PREWITT_Y = np.array([[-1, -1, -1],
                      [0, 0, 0],
                      [1, 1, 1]], dtype=np.float32)

# Sobel: like Prewitt but the centre row/column is weighted 2, which gives
# a mild Gaussian smoothing and therefore better noise rejection.
SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)
SOBEL_Y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]], dtype=np.float32)

# Scharr: a more rotationally accurate 3x3 alternative to Sobel.
SCHARR_X = np.array([[-3, 0, 3],
                     [-10, 0, 10],
                     [-3, 0, 3]], dtype=np.float32)
SCHARR_Y = np.array([[-3, -10, -3],
                     [0, 0, 0],
                     [3, 10, 3]], dtype=np.float32)


def _gradient_pair(image, kernel_x, kernel_y):
    """Convolve the grayscale image with both kernels, keeping float output
    so that negative gradient values are not clipped away.
    """
    gray = ensure_gray(image).astype(np.float32)
    gx = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
    gy = cv2.filter2D(gray, cv2.CV_32F, kernel_y)
    return gx, gy


def gradient_magnitude(gx, gy):
    """magnitude = sqrt(Gx^2 + Gy^2)  (the true Euclidean gradient norm)."""
    return np.sqrt(np.square(gx.astype(np.float32)) + np.square(gy.astype(np.float32)))


def gradient_direction(gx, gy, degrees=True):
    """direction = arctan2(Gy, Gx) - the angle of steepest intensity change,
    perpendicular to the edge itself.
    """
    angle = np.arctan2(gy.astype(np.float32), gx.astype(np.float32))
    return np.degrees(angle) if degrees else angle


def roberts_operator(image):
    """Roberts cross-gradient. Returns dict with x, y, magnitude (uint8)."""
    gx, gy = _gradient_pair(image, ROBERTS_X, ROBERTS_Y)
    return _package(gx, gy)


def prewitt_operator(image):
    """Prewitt operator. Returns dict with x, y, magnitude (uint8)."""
    gx, gy = _gradient_pair(image, PREWITT_X, PREWITT_Y)
    return _package(gx, gy)


def sobel_operator(image, kernel_size=3):
    """Sobel operator via cv2.Sobel (kernel size 3, 5 or 7).

    Returns dict with x, y, magnitude (uint8) plus the raw float gradients.
    """
    gray = ensure_gray(image).astype(np.float32)
    kernel_size = int(kernel_size)
    if kernel_size not in (1, 3, 5, 7):
        kernel_size = 3
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=kernel_size)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=kernel_size)
    return _package(gx, gy)


def scharr_operator(image):
    """Scharr operator - optimised 3x3 kernel with better rotational symmetry."""
    gray = ensure_gray(image).astype(np.float32)
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    return _package(gx, gy)


def _package(gx, gy):
    """Bundle raw + display-ready versions of a gradient pair."""
    magnitude = gradient_magnitude(gx, gy)
    direction = gradient_direction(gx, gy)
    return {
        "gx_raw": gx,
        "gy_raw": gy,
        "magnitude_raw": magnitude,
        "direction_raw": direction,
        "x": normalize_to_uint8(np.abs(gx)),
        "y": normalize_to_uint8(np.abs(gy)),
        "magnitude": normalize_to_uint8(magnitude),
    }


def direction_visualization(magnitude, direction):
    """Colour-code the gradient direction.

    Hue  = edge orientation (0-360 degrees mapped to the colour wheel)
    Value = gradient magnitude, so flat areas stay black.

    This is a compact alternative to drawing a vector field.
    """
    hue = ((direction % 360.0) / 2.0).astype(np.uint8)          # OpenCV hue is 0-179
    saturation = np.full(hue.shape, 255, dtype=np.uint8)
    value = normalize_to_uint8(magnitude)
    hsv = cv2.merge([hue, saturation, value])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def all_operators(image, sobel_kernel=3):
    """Run Roberts, Prewitt and Sobel at once for the comparison grid."""
    return {
        "Roberts": roberts_operator(image),
        "Prewitt": prewitt_operator(image),
        "Sobel": sobel_operator(image, sobel_kernel),
    }
