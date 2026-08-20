"""Noise removal and image smoothing.

Smoothing filters replace each pixel with some statistic of its
neighbourhood.  Which statistic you choose decides which noise you
suppress and how much edge detail you lose.
"""

import cv2
import numpy as np

from .utils import ensure_color, odd_kernel


def gaussian_blur(image, kernel_size=5, sigma=1.0):
    """Weighted average using a Gaussian kernel.

    Nearby pixels count more than distant ones, so edges are preserved a
    little better than with a plain box filter.  Best against Gaussian
    (sensor / thermal) noise.  sigma = 0 lets OpenCV derive it from the
    kernel size.
    """
    kernel_size = odd_kernel(kernel_size, minimum=1)
    return cv2.GaussianBlur(np.asarray(image), (kernel_size, kernel_size), float(sigma))


def median_filter(image, kernel_size=3):
    """Replace each pixel with the MEDIAN of its neighbourhood.

    A non-linear order-statistic filter.  Because an extreme value cannot
    drag the median, it removes salt-and-pepper (impulse) noise almost
    perfectly while keeping edges sharp.
    """
    kernel_size = odd_kernel(kernel_size, minimum=3)
    return cv2.medianBlur(np.asarray(image).astype(np.uint8), kernel_size)


def average_filter(image, kernel_size=3):
    """Mean / box filter - every neighbour gets the same weight 1/(k*k).

    The simplest low-pass filter.  Cheap, but it blurs edges as strongly
    as it blurs noise.
    """
    kernel_size = odd_kernel(kernel_size, minimum=1)
    return cv2.blur(np.asarray(image), (kernel_size, kernel_size))


def bilateral_filter(image, diameter=9, sigma_color=75, sigma_space=75):
    """Edge-preserving smoothing.

    Weights neighbours by BOTH spatial distance and intensity difference,
    so pixels across an edge barely contribute.  Result: flat regions get
    smoothed, edges stay crisp.
    """
    array = np.asarray(image).astype(np.uint8)
    if array.ndim == 3 and array.shape[2] != 3:
        # OpenCV only accepts 1- or 3-channel input here (an RGBA image
        # would raise), so drop the alpha channel first.
        array = ensure_color(array)
    return cv2.bilateralFilter(
        array, int(diameter), float(sigma_color), float(sigma_space)
    )


def add_gaussian_noise(image, mean=0.0, sigma=20.0, seed=0):
    """Add synthetic Gaussian noise - useful for demonstrating that the
    Gaussian / mean filters are the right tool for this noise model.
    """
    array = np.asarray(image).astype(np.float32)
    rng = np.random.default_rng(int(seed))
    noise = rng.normal(float(mean), float(sigma), array.shape)
    return np.clip(array + noise, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(image, amount=0.05, seed=0):
    """Add salt-and-pepper (impulse) noise: a fraction of pixels is forced
    to pure white or pure black.  The classic use-case for a median filter.
    """
    array = np.asarray(image).astype(np.uint8).copy()
    rng = np.random.default_rng(int(seed))
    height, width = array.shape[:2]
    total = int(np.clip(amount, 0.0, 1.0) * height * width)
    if total == 0:
        return array
    for value in (255, 0):  # salt, then pepper
        ys = rng.integers(0, height, total // 2)
        xs = rng.integers(0, width, total // 2)
        array[ys, xs] = value
    return array


def denoise_metrics(original, processed):
    """Report MSE and PSNR so filters can be compared numerically.

    PSNR = 10 * log10(255^2 / MSE); higher means closer to the reference.
    """
    a = np.asarray(original).astype(np.float64)
    b = np.asarray(processed).astype(np.float64)
    if a.shape != b.shape:
        return None
    mse = float(np.mean((a - b) ** 2))
    if mse < 1e-9:
        return {"mse": 0.0, "psnr": float("inf")}
    return {"mse": mse, "psnr": float(10.0 * np.log10((255.0 ** 2) / mse))}
