"""Spatial filtering: low-pass, high-pass and custom convolution kernels.

A spatial filter slides a small kernel (mask) over the image and replaces
each pixel with a weighted sum of its neighbourhood.  The kernel weights
alone decide whether the filter smooths (low-pass) or sharpens /
detects edges (high-pass).
"""

import cv2
import numpy as np

from .utils import normalize_to_uint8, odd_kernel

# --- Standard kernels shown in the UI ---------------------------------------

# High-pass: coefficients sum to 0, so flat regions -> 0 and only rapid
# intensity changes survive.
HIGH_PASS_KERNEL = np.array(
    [[-1, -1, -1],
     [-1, 8, -1],
     [-1, -1, -1]], dtype=np.float32
)

HIGH_PASS_KERNEL_SOFT = np.array(
    [[0, -1, 0],
     [-1, 4, -1],
     [0, -1, 0]], dtype=np.float32
)

SHARPEN_KERNEL = np.array(
    [[0, -1, 0],
     [-1, 5, -1],
     [0, -1, 0]], dtype=np.float32
)

EMBOSS_KERNEL = np.array(
    [[-2, -1, 0],
     [-1, 1, 1],
     [0, 1, 2]], dtype=np.float32
)


def apply_kernel(image, kernel, normalize=False):
    """Convolve `image` with `kernel` (cv2.filter2D does correlation, which
    is identical for the symmetric kernels used here).

    normalize=True rescales a signed result into 0-255 for display; use it
    for high-pass kernels whose output contains negative values.
    """
    kernel = np.asarray(kernel, dtype=np.float32)
    if normalize:
        filtered = cv2.filter2D(np.asarray(image).astype(np.float32), cv2.CV_32F, kernel)
        return normalize_to_uint8(filtered)
    return cv2.filter2D(np.asarray(image), -1, kernel)


def average_kernel(kernel_size=3):
    """Build a normalised k x k averaging (box) kernel."""
    kernel_size = odd_kernel(kernel_size, minimum=1)
    return np.ones((kernel_size, kernel_size), np.float32) / float(kernel_size ** 2)


def low_pass_average(image, kernel_size=3):
    """Low-pass filtering with a box kernel - attenuates high frequencies
    (fine detail and noise) and keeps the slowly varying content.
    """
    return cv2.blur(np.asarray(image), (odd_kernel(kernel_size),) * 2)


def low_pass_gaussian(image, kernel_size=5, sigma=1.0):
    """Low-pass filtering with a Gaussian kernel - a smoother frequency
    roll-off than the box filter, so it produces fewer ringing artefacts.
    """
    kernel_size = odd_kernel(kernel_size, minimum=1)
    return cv2.GaussianBlur(np.asarray(image), (kernel_size, kernel_size), float(sigma))


def high_pass_filter(image, strong=True):
    """High-pass filtering - emphasises rapid intensity changes (edges,
    texture, fine detail) and removes the smooth background.
    """
    kernel = HIGH_PASS_KERNEL if strong else HIGH_PASS_KERNEL_SOFT
    return apply_kernel(image, kernel, normalize=True)


def high_pass_by_subtraction(image, kernel_size=9, sigma=0.0):
    """The other way to build a high-pass filter:

        high_pass = original - low_pass(original)

    Shows explicitly that a high-pass filter is just the complement of a
    low-pass one.
    """
    array = np.asarray(image).astype(np.float32)
    kernel_size = odd_kernel(kernel_size, minimum=3)
    low = cv2.GaussianBlur(array, (kernel_size, kernel_size), float(sigma))
    return normalize_to_uint8(array - low)


def sharpen_filter(image):
    """Practical high-frequency enhancement: original + high-pass detail."""
    return cv2.filter2D(np.asarray(image), -1, SHARPEN_KERNEL)


def emboss_filter(image):
    """Directional high-pass filter that gives a 3-D relief appearance."""
    return apply_kernel(image, EMBOSS_KERNEL, normalize=True)
