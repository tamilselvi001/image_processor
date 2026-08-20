"""Edge detection operators.

First-derivative operators (Roberts, Prewitt, Sobel) mark an edge where
the gradient magnitude is a MAXIMUM.  Second-derivative operators
(Laplacian, LoG) mark an edge where the second derivative crosses ZERO.
"""

import cv2
import numpy as np

from . import gradients
from .utils import ensure_gray, normalize_to_uint8, odd_kernel

# 4-neighbour and 8-neighbour Laplacian masks.  Both sum to zero.
LAPLACIAN_4 = np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]], dtype=np.float32)

LAPLACIAN_8 = np.array([[1, 1, 1],
                        [1, -8, 1],
                        [1, 1, 1]], dtype=np.float32)


def _to_binary(edge_map, threshold):
    """Optionally convert a grayscale edge strength map into a binary map."""
    if threshold is None:
        return edge_map
    _, binary = cv2.threshold(edge_map, int(threshold), 255, cv2.THRESH_BINARY)
    return binary


def roberts_edges(image, threshold=None):
    """Roberts cross-gradient edge map (magnitude of the 2x2 diagonal diffs)."""
    return _to_binary(gradients.roberts_operator(image)["magnitude"], threshold)


def prewitt_edges(image, threshold=None):
    """Prewitt edge map."""
    return _to_binary(gradients.prewitt_operator(image)["magnitude"], threshold)


def sobel_edges(image, kernel_size=3, threshold=None):
    """Sobel edge map."""
    return _to_binary(gradients.sobel_operator(image, kernel_size)["magnitude"], threshold)


def laplacian_operator(image, kernel_size=3, threshold=None):
    """Laplacian: the sum of the second derivatives,  L = d2I/dx2 + d2I/dy2.

    It is isotropic (no preferred direction) but, being a second
    derivative, it is very sensitive to noise - which is exactly why the
    Laplacian of Gaussian exists.
    """
    gray = ensure_gray(image)
    kernel_size = odd_kernel(kernel_size, minimum=1)
    laplacian = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=kernel_size)
    return _to_binary(normalize_to_uint8(np.abs(laplacian)), threshold)


def laplacian_of_gaussian(image, kernel_size=5, sigma=1.4, threshold=None):
    """Laplacian of Gaussian (LoG / Marr-Hildreth):

        1. smooth with a Gaussian to suppress noise
        2. apply the Laplacian
        3. edges are the zero-crossings of the result
    """
    gray = ensure_gray(image).astype(np.float32)
    kernel_size = odd_kernel(kernel_size, minimum=3)
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), float(sigma))
    log = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
    return _to_binary(normalize_to_uint8(np.abs(log)), threshold)


def zero_crossing(image, kernel_size=5, sigma=1.4):
    """Explicit zero-crossing detector for the LoG response.

    A pixel is an edge if the LoG changes sign between opposite
    neighbours - that sign change is the actual edge location.
    """
    gray = ensure_gray(image).astype(np.float32)
    kernel_size = odd_kernel(kernel_size, minimum=3)
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), float(sigma))
    log = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)

    minimum = cv2.erode(log, np.ones((3, 3), np.uint8))
    maximum = cv2.dilate(log, np.ones((3, 3), np.uint8))
    crossing = ((minimum < 0) & (log > 0)) | ((maximum > 0) & (log < 0))
    return (crossing.astype(np.uint8)) * 255


def compare_edge_detectors(image, sobel_kernel=3, log_sigma=1.4, threshold=None):
    """Run every operator once so the UI can show them side by side."""
    return {
        "Roberts": roberts_edges(image, threshold),
        "Prewitt": prewitt_edges(image, threshold),
        "Sobel": sobel_edges(image, sobel_kernel, threshold),
        "Laplacian": laplacian_operator(image, 3, threshold),
        "Laplacian of Gaussian": laplacian_of_gaussian(image, 5, log_sigma, threshold),
    }
