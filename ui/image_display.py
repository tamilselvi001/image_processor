"""Image presentation helpers: side-by-side comparison, grids, histograms,
before/after wipe and PNG/JPG downloads.

Nothing here modifies the arrays it is given.
"""

import io

import cv2
import matplotlib
import numpy as np
import streamlit as st

matplotlib.use("Agg")  # headless backend - required inside Streamlit
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from processors.utils import ensure_color, ensure_gray  # noqa: E402

from . import theme  # noqa: E402


# --------------------------------------------------------------------------
# Encoding / download
# --------------------------------------------------------------------------

def to_pil(image):
    """numpy array -> PIL Image (handles 2-D grayscale and 3-D RGB)."""
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        return Image.fromarray(array, mode="L")
    if array.shape[2] == 4:
        return Image.fromarray(array, mode="RGBA")
    return Image.fromarray(array, mode="RGB")


def to_bytes(image, fmt="PNG"):
    """Encode an image to in-memory bytes - nothing is written to disk."""
    buffer = io.BytesIO()
    pil_image = to_pil(image)
    if fmt.upper() in ("JPG", "JPEG"):
        pil_image.convert("RGB").save(buffer, format="JPEG", quality=95)
    else:
        pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def download_image(image, filename, key, label="Download Processed Image"):
    """Download button offering PNG and JPG for the processed result."""
    left, right = st.columns(2)
    stem = filename.rsplit(".", 1)[0]
    with left:
        st.download_button(
            label,
            data=to_bytes(image, "PNG"),
            file_name=f"{stem}.png",
            mime="image/png",
            key=f"{key}_png",
            width="stretch",
        )
    with right:
        st.download_button(
            "Download as JPG",
            data=to_bytes(image, "JPEG"),
            file_name=f"{stem}.jpg",
            mime="image/jpeg",
            key=f"{key}_jpg",
            width="stretch",
        )


# --------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------

def panel(image, caption):
    """A labelled image panel."""
    st.markdown(f'<div class="iva-imgcap">{caption}</div>', unsafe_allow_html=True)
    st.image(image, width="stretch")


def compare(original, processed, original_label="Original",
            processed_label="Processed"):
    """Standard two-column Original | Processed view."""
    left, right = st.columns(2)
    with left:
        panel(original, original_label)
    with right:
        panel(processed, processed_label)


def grid(images, columns=3):
    """Render a {caption: image} mapping as an even grid."""
    items = list(images.items())
    columns = max(1, int(columns))
    for start in range(0, len(items), columns):
        row = st.columns(columns)
        for column, (caption, image) in zip(row, items[start:start + columns]):
            with column:
                panel(image, caption)


def before_after_slider(original, processed, key, label="Wipe position (%)"):
    """A simple before/after wipe.

    The output is one image whose left part comes from the original and
    whose right part comes from the processed result; the slider moves the
    split.  Implemented with plain array slicing - no custom component.
    """
    base = ensure_color(original)
    result = ensure_color(processed)

    if result.shape[:2] != base.shape[:2]:
        result = cv2.resize(result, (base.shape[1], base.shape[0]),
                            interpolation=cv2.INTER_AREA)

    position = st.slider(label, 0, 100, 50, key=key)
    split = int(base.shape[1] * position / 100.0)

    composite = result.copy()
    if split > 0:
        composite[:, :split] = base[:, :split]

    # Thin divider line so the split position is obvious.
    if 0 < split < base.shape[1]:
        line_end = min(split + 2, base.shape[1])
        composite[:, split:line_end] = np.array([230, 230, 230], dtype=np.uint8)

    st.markdown(
        '<div class="iva-imgcap">Before (left) / After (right)</div>',
        unsafe_allow_html=True,
    )
    st.image(composite, width="stretch")


# --------------------------------------------------------------------------
# Histograms
# --------------------------------------------------------------------------

def _style_axes(ax, style, title):
    ax.set_facecolor(style["face"])
    ax.set_title(title, color=style["text"], fontsize=10, pad=8)
    ax.set_xlabel("Intensity (0-255)", color=style["muted"], fontsize=8)
    ax.set_ylabel("Pixel count", color=style["muted"], fontsize=8)
    ax.tick_params(colors=style["muted"], labelsize=7)
    ax.grid(True, color=style["grid"], linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(style["border"])
    ax.set_xlim(0, 255)


def histogram_figure(image, title="Grayscale Histogram"):
    """Matplotlib grayscale histogram styled to match the active theme."""
    style = theme.matplotlib_style()
    gray = ensure_gray(image)

    fig, ax = plt.subplots(figsize=(5.2, 2.5), dpi=130)
    fig.patch.set_facecolor(style["face"])
    ax.hist(gray.ravel(), bins=256, range=(0, 256),
            color=style["accent"], alpha=0.85, linewidth=0)
    _style_axes(ax, style, title)
    fig.tight_layout()
    return fig


def rgb_histogram_figure(image, title="RGB Channel Histogram"):
    """Per-channel histogram for colour images."""
    style = theme.matplotlib_style()
    color_image = ensure_color(image)

    fig, ax = plt.subplots(figsize=(5.2, 2.5), dpi=130)
    fig.patch.set_facecolor(style["face"])
    for index, (name, color) in enumerate(
        (("Red", "#d1615d"), ("Green", "#6a9a5b"), ("Blue", "#5b83b8"))
    ):
        counts, _ = np.histogram(color_image[:, :, index].ravel(),
                                 bins=256, range=(0, 256))
        ax.plot(np.arange(256), counts, color=color, linewidth=1.0, label=name)
    _style_axes(ax, style, title)
    legend = ax.legend(fontsize=7, framealpha=0.0)
    for text in legend.get_texts():
        text.set_color(style["muted"])
    fig.tight_layout()
    return fig


def show_figure(fig):
    """Render and immediately close a Matplotlib figure (avoids leaks)."""
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def histogram_pair(original, processed=None,
                   original_title="Original Histogram",
                   processed_title="Processed Histogram"):
    """Original (and optionally processed) histograms side by side."""
    if processed is None:
        show_figure(histogram_figure(original, original_title))
        return
    left, right = st.columns(2)
    with left:
        show_figure(histogram_figure(original, original_title))
    with right:
        show_figure(histogram_figure(processed, processed_title))


def intensity_stats(image):
    """Min / max / mean / std of the grayscale intensities."""
    gray = ensure_gray(image).astype(np.float64)
    return {
        "min": int(gray.min()),
        "max": int(gray.max()),
        "mean": round(float(gray.mean()), 1),
        "std": round(float(gray.std()), 1),
    }
