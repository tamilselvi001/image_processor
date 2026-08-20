"""Image-processing operations for Image Processing Studio.

Each module covers one family of operations:

    spatial.py        point operations (grayscale, negative, thresholding, ...)
    enhancement.py    brightness, contrast, gamma, sharpening
    noise_removal.py  Gaussian / median / mean / bilateral smoothing
    filters.py        low-pass, high-pass and custom convolution kernels
    gradients.py      Roberts, Prewitt, Sobel; magnitude and direction
    edge_detection.py Roberts/Prewitt/Sobel/Laplacian/LoG edge maps
    canny.py          the multi-stage Canny detector
    shapes.py         contours and classical shape recognition
    utils.py          shared colour-space and normalisation helpers

Every public function returns a NEW array - the uploaded image is never
modified in place.
"""

from . import (  # noqa: F401
    canny,
    edge_detection,
    enhancement,
    filters,
    gradients,
    noise_removal,
    shapes,
    spatial,
    utils,
)

__all__ = [
    "spatial",
    "enhancement",
    "noise_removal",
    "filters",
    "gradients",
    "edge_detection",
    "canny",
    "shapes",
    "utils",
]
