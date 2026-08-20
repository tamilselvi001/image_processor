"""Image enhancement: brightness, contrast, gamma and sharpening.

Enhancement makes an image more suitable for a *specific* purpose -
there is no single "best" enhanced image, only one that shows the
feature you care about more clearly.
"""

import cv2
import numpy as np

# The classic 3x3 sharpening kernel (a Laplacian high-pass core plus the
# original image).  Its coefficients sum to 1, so average brightness is kept.
SHARPEN_KERNEL = np.array(
    [[0, -1, 0],
     [-1, 5, -1],
     [0, -1, 0]], dtype=np.float32
)


def adjust_brightness(image, beta=0):
    """Add a constant to every pixel:  output = input + beta."""
    return cv2.convertScaleAbs(np.asarray(image), alpha=1.0, beta=float(beta))


def adjust_contrast(image, alpha=1.0):
    """Scale every pixel about zero:  output = alpha * input."""
    return cv2.convertScaleAbs(np.asarray(image), alpha=float(alpha), beta=0.0)


def brightness_contrast(image, alpha=1.0, beta=0):
    """Combined linear transform:  output = alpha * input + beta.

    alpha controls contrast (gain), beta controls brightness (bias).
    convertScaleAbs saturates at 0/255 so values never wrap around.
    """
    return cv2.convertScaleAbs(np.asarray(image), alpha=float(alpha), beta=float(beta))


def gamma_correction(image, gamma=1.0):
    """Non-linear intensity mapping:  output = 255 * (input / 255) ** (1 / gamma).

    gamma > 1 brightens the mid-tones, gamma < 1 darkens them.  A 256-entry
    lookup table is built once and applied to the whole image, which is far
    faster than computing the power per pixel.
    """
    gamma = float(gamma)
    if gamma <= 0:
        gamma = 0.1
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8
    )
    return cv2.LUT(np.asarray(image).astype(np.uint8), table)


def sharpen(image, kernel=None):
    """Sharpen with the 3x3 [[0,-1,0],[-1,5,-1],[0,-1,0]] kernel.

    It is 'original + high-pass detail', so edges gain local contrast.
    """
    kernel = SHARPEN_KERNEL if kernel is None else np.asarray(kernel, dtype=np.float32)
    return cv2.filter2D(np.asarray(image), -1, kernel)


def unsharp_mask(image, kernel_size=5, sigma=1.0, amount=1.0):
    """Unsharp masking:  sharp = original + amount * (original - blurred).

    The blurred copy is a low-pass version, so the difference is the
    high-frequency detail that gets amplified.
    """
    array = np.asarray(image)
    kernel_size = int(kernel_size) | 1  # force odd
    blurred = cv2.GaussianBlur(array, (kernel_size, kernel_size), float(sigma))
    sharpened = cv2.addWeighted(
        array.astype(np.float32), 1.0 + float(amount),
        blurred.astype(np.float32), -float(amount), 0.0,
    )
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def laplacian_sharpen(image, strength=1.0):
    """Laplacian sharpening:  output = original - strength * Laplacian(original).

    The Laplacian is a second-derivative operator: it is strongly negative
    on the bright side of an edge and positive on the dark side, so
    subtracting it exaggerates the transition.
    """
    array = np.asarray(image)
    laplacian = cv2.Laplacian(array, cv2.CV_32F, ksize=3)
    sharpened = array.astype(np.float32) - float(strength) * laplacian
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def clahe(image, clip_limit=2.0, tile_grid=8):
    """Contrast Limited Adaptive Histogram Equalisation.

    Equalises small tiles independently and clips the histogram to stop
    noise from being over-amplified - a gentler alternative to global
    histogram equalisation.
    """
    array = np.asarray(image)
    tile = max(1, int(tile_grid))
    engine = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(tile, tile))
    if array.ndim == 2:
        return engine.apply(array)
    lab = cv2.cvtColor(array, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = engine.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
