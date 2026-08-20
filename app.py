"""Image Processing — standalone classical computer vision tool.

Run with:   streamlit run app.py

This file holds only the Streamlit UI and page routing.  All of the
actual image processing lives in the `processors` package, and all of
the presentation helpers live in the `ui` package.
"""

import os

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

from processors import (
    canny as canny_ops,
    edge_detection,
    enhancement,
    filters,
    gradients,
    noise_removal,
    shapes,
    spatial,
)
from processors.utils import ensure_color, ensure_gray, image_info, resize_for_processing
from ui import components as ui
from ui import image_display as disp
from ui import theme as ui_theme

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

PAGES = [
    "Home",
    "Image Processing",
    "Enhancement",
    "Filters",
    "Gradients",
    "Edge Detection",
    "Canny",
    "Shape Detection",
]

SAMPLES = {
    "Geometric Shapes": "sample_shapes.png",
    "Test Pattern": "sample_pattern.png",
}


# ==========================================================================
# Session state / image handling
# ==========================================================================

def init_state():
    defaults = {
        "theme": "dark",
        "page": "Home",
        "original": None,
        "filename": None,
        "source_shape": None,
        "was_resized": False,
        "max_dim": 1200,
        "upload_round": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def get_image():
    """The working copy of the uploaded image (RGB or grayscale uint8)."""
    return st.session_state.get("original")


def has_image():
    return st.session_state.get("original") is not None


def store_image(array, filename):
    """Validate, optionally downscale and store the uploaded image."""
    array = np.asarray(array)
    if array.size == 0 or array.ndim not in (2, 3):
        st.error("That file could not be read as an image.")
        return False
    if array.ndim == 3 and array.shape[2] not in (3, 4):
        st.error("Unsupported channel count - please upload an RGB or grayscale image.")
        return False

    source_shape = array.shape
    working, was_resized = resize_for_processing(array, st.session_state["max_dim"])

    st.session_state["original"] = working
    st.session_state["filename"] = filename
    st.session_state["source_shape"] = source_shape
    st.session_state["was_resized"] = was_resized
    return True


def load_uploaded_file(uploaded_file):
    """Decode an uploaded file with Pillow, converting to RGB or grayscale."""
    try:
        pil_image = Image.open(uploaded_file)
        pil_image.load()
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("Invalid or corrupted image file. Try a different JPG, PNG or BMP.")
        return None

    if pil_image.mode in ("L", "1", "I;16"):
        pil_image = pil_image.convert("L")
    elif pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    return np.array(pil_image)


def clear_image():
    """Unload the current image.

    The uploader's key is rotated as well, otherwise the widget would still
    be holding the previous file and would immediately re-load it on the
    next rerun.
    """
    st.session_state["original"] = None
    st.session_state["filename"] = None
    st.session_state["source_shape"] = None
    st.session_state["was_resized"] = False
    st.session_state["upload_round"] += 1


def download_name(suffix):
    """Build a sensible download filename from the uploaded file name."""
    base = st.session_state.get("filename") or "image"
    stem = os.path.splitext(os.path.basename(base))[0]
    return f"{stem}_{suffix}.png"


# ==========================================================================
# Sidebar
# ==========================================================================

def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div style="font-size:.95rem;font-weight:650;letter-spacing:-.01em;'
            'margin-bottom:.1rem;">Image Processing </div>'
            '<div style="font-size:.72rem;color:var(--iva-muted);'
            'margin-bottom:1rem;">Classical computer vision tools</div>',
            unsafe_allow_html=True,
        )

        # --- Theme toggle ---
        is_dark = st.session_state["theme"] == "dark"
        label = "Switch to Light Theme" if is_dark else "Switch to Dark Theme"
        if st.button(label, key="theme_toggle", width="stretch"):
            ui_theme.toggle_theme()
            st.rerun()

        st.markdown(
            '<div style="height:1px;background:var(--iva-border);'
            'margin:1rem 0;"></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div style="font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;'
            'color:var(--iva-muted);margin-bottom:.4rem;">Tools</div>',
            unsafe_allow_html=True,
        )
        st.radio("Navigation", PAGES, key="page", label_visibility="collapsed")

        st.markdown(
            '<div style="height:1px;background:var(--iva-border);'
            'margin:1rem 0;"></div>',
            unsafe_allow_html=True,
        )

        # --- Global image loader ---
        st.markdown(
            '<div style="font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;'
            'color:var(--iva-muted);margin-bottom:.4rem;">Image</div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Choose image", type=["jpg", "jpeg", "png", "bmp"],
            key=f"global_uploader_{st.session_state['upload_round']}",
            label_visibility="collapsed",
        )
        if uploaded is not None and st.session_state.get("filename") != uploaded.name:
            array = load_uploaded_file(uploaded)
            if array is not None and store_image(array, uploaded.name):
                st.rerun()

        # --- Current image summary ---
        if has_image():
            info = image_info(get_image())
            st.markdown(
                '<div style="font-size:.7rem;text-transform:uppercase;'
                'letter-spacing:.08em;color:var(--iva-muted);'
                'margin-bottom:.4rem;">Current Image</div>',
                unsafe_allow_html=True,
            )
            st.image(get_image(), width="stretch")
            st.markdown(
                f'<div style="font-size:.73rem;color:var(--iva-muted);'
                f'line-height:1.6;margin-top:.4rem;">'
                f'{st.session_state["filename"]}<br>'
                f'{info["width"]} &times; {info["height"]} px &middot; '
                f'{info["channels"]} channel(s)</div>',
                unsafe_allow_html=True,
            )
            if st.button("Reset Image", key="sidebar_reset", width="stretch"):
                clear_image()
                st.rerun()
        else:
            st.markdown(
                '<div style="font-size:.76rem;color:var(--iva-muted);'
                'line-height:1.6;">No image loaded yet. Go to '
                '<b>Image Processing</b> to begin.</div>',
                unsafe_allow_html=True,
            )


# ==========================================================================
# Pages
# ==========================================================================

def page_home():
    ui.section_header(
        "Image Processing ",
        "Explore, enhance and analyze images using practical classical computer-vision tools.",
    )

    ui.note(
        "Runs locally with OpenCV and NumPy. Your image stays in memory and is never uploaded to a server."
    )

    cols = st.columns(3)
    cards = [
        ("Image Processing", "Pixel-level transformations, thresholding and histogram tools."),
        ("Enhancement", "Brightness, contrast, gamma and sharpening controls."),
        ("Filters", "Smoothing, noise reduction and spatial filtering."),
        ("Gradients", "Roberts, Prewitt, Sobel, magnitude and direction."),
        ("Edge Detection", "Roberts, Prewitt, Sobel, Laplacian and LoG."),
        ("Canny", "Adjustable Canny edge detection with clean result preview."),
        ("Shape Detection", "Contours and basic geometric shape detection."),
    ]
    for i, (title, desc) in enumerate(cards):
        with cols[i % 3]:
            st.markdown(
                f'<div class="iva-card" style="margin-bottom:1rem;min-height:112px;">'
                f'<div class="iva-card .iva-label"></div>'
                f'<div style="font-size:1rem;font-weight:620;color:var(--iva-text);margin-bottom:.35rem;">{title}</div>'
                f'<div style="font-size:.8rem;color:var(--iva-muted);line-height:1.5;">{desc}</div>'
                f'</div>', unsafe_allow_html=True
            )

    if has_image():
        st.markdown("#### Current Image")
        disp.panel(get_image(), "Original")
    else:
        st.markdown("#### Get Started")
        st.caption("Use the image picker in the sidebar to load an image, then select any tool.")

def page_image_processing():
    ui.section_header(
        "Image Processing",
        "Load an image and access the processing tools available in the studio.",
    )

    left, right = st.columns([1.4, 1])

    with left:
        uploaded = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "bmp"],
            key=f"uploader_{st.session_state['upload_round']}",
        )
        if uploaded is not None:
            array = load_uploaded_file(uploaded)
            if array is not None and st.session_state.get("filename") != uploaded.name:
                if store_image(array, uploaded.name):
                    st.rerun()

    with right:
        st.markdown(
            '<div style="font-size:.8rem;color:var(--iva-muted);margin-bottom:.4rem;">'
            "Or load a built-in sample</div>",
            unsafe_allow_html=True,
        )
        for name, file_name in SAMPLES.items():
            path = os.path.join(ASSETS_DIR, file_name)
            if st.button(name, key=f"sample_{file_name}", width="stretch"):
                if os.path.exists(path):
                    store_image(np.array(Image.open(path).convert("RGB")), file_name)
                    st.rerun()
                else:
                    st.error(f"Sample file not found: {file_name}")

        st.session_state["max_dim"] = st.select_slider(
            "Max image size (longest side)",
            options=[600, 800, 1000, 1200, 1600, 2000],
            value=st.session_state["max_dim"],
            key="max_dim_widget",
        )
        st.caption("Applies when a new image is loaded.")

    ui.note(
        "Large images may be resized for faster processing. The aspect ratio is "
        "always preserved, and the resize happens once at upload time."
    )

    if not has_image():
        ui.no_image_notice()
        return

    ui.divider()

    image = get_image()
    info = image_info(image)

    st.markdown("#### Original Image")
    ui.image_info_cards(info)
    st.write("")

    if st.session_state.get("was_resized"):
        source = st.session_state.get("source_shape")
        ui.warn(
            f"This image was resized for processing from "
            f"{source[1]} &times; {source[0]} px to "
            f"{info['width']} &times; {info['height']} px "
            "(aspect ratio preserved)."
        )

    image_column, hist_column = st.columns([1.25, 1])
    with image_column:
        disp.panel(image, "Original")
    with hist_column:
        disp.show_figure(disp.histogram_figure(image, "Grayscale Histogram"))
        if image.ndim == 3:
            disp.show_figure(disp.rgb_histogram_figure(image))

    stats = disp.intensity_stats(image)
    ui.stat_row([
        ("Min Intensity", stats["min"]),
        ("Max Intensity", stats["max"]),
        ("Mean", stats["mean"]),
        ("Std Deviation", stats["std"]),
        ("Total Pixels", f"{info['pixels']:,}"),
    ])

    ui.divider()
    left_button, right_button = st.columns([1, 3])
    with left_button:
        if st.button("Reset Image", key="upload_reset", width="stretch"):
            clear_image()
            st.rerun()


# --------------------------------------------------------------------------

def page_spatial():
    ui.section_header(
        "Image Processing",
        "Apply pixel-level transformations, thresholding, histograms and spatial filters to the current image. "

    )

    if not has_image():
        ui.no_image_notice()
        return

    image = get_image()

    tabs = st.tabs([
        "Grayscale", "Negative", "Thresholding", "Contrast Stretching",
        "Histogram", "Spatial Filters",
    ])

    # ---- Grayscale ----
    with tabs[0]:
        ui.concept(
            "Grayscale Conversion",
            "A colour image is collapsed into a single intensity channel using the "
            "luminosity weights, which match how sensitive the human eye is to each "
            "primary colour. Almost every edge and gradient operator expects a "
            "single-channel input, so this is usually step one.",
            formula="Y = 0.299 R + 0.587 G + 0.114 B",
        )
        gray = spatial.to_grayscale(image)
        disp.compare(image, gray, "Original", "Grayscale")
        disp.download_image(gray, download_name("grayscale"), "dl_gray")

    # ---- Negative ----
    with tabs[1]:
        ui.concept(
            "Image Negative",
            "Every intensity is inverted about the middle of the range. It is the "
            "simplest possible point operation and is genuinely useful for spotting "
            "detail hidden inside dark regions, such as in X-ray imagery.",
            formula="output = 255 - input",
        )
        col_a, col_b = st.columns([1, 3])
        with col_a:
            apply_negative = st.button("Apply Negative", key="neg_apply",
                                       width="stretch")
        if apply_negative or st.session_state.get("neg_shown"):
            st.session_state["neg_shown"] = True
            result = spatial.negative(image)
            disp.compare(image, result, "Original", "Negative")
            disp.histogram_pair(image, result, "Original Histogram",
                                "Negative Histogram")
            disp.download_image(result, download_name("negative"), "dl_neg")
            with col_b:
                if st.button("Restore Original", key="neg_reset"):
                    st.session_state.pop("neg_shown", None)
                    st.rerun()

    # ---- Thresholding ----
    with tabs[2]:
        ui.concept(
            "Thresholding",
            "Thresholding segments an image into foreground and background by "
            "comparing each pixel with a fixed value. Otsu's method removes the "
            "guesswork by choosing the threshold that best separates the two "
            "intensity populations in the histogram.",
            formula="output(x, y) = 255  if input(x, y) > T\n"
                    "             = 0    otherwise",
        )
        controls, preview = st.columns([1, 2.4])
        with controls:
            threshold = st.slider("Threshold value", 0, 255, 127, key="thr_value")
            invert = st.checkbox("Invert (THRESH_BINARY_INV)", key="thr_invert")
            use_otsu = st.button("Apply Otsu Thresholding", key="thr_otsu",
                                 width="stretch")
            use_adaptive = st.checkbox("Use adaptive thresholding", key="thr_adaptive")
            if use_adaptive:
                block = st.slider("Block size", 3, 51, 11, step=2, key="thr_block")
                constant = st.slider("Constant C", -10, 20, 2, key="thr_const")
            ui.reset_button(
                ["thr_value", "thr_invert", "thr_adaptive", "thr_block", "thr_const"],
                key="thr_reset",
            )

        if use_adaptive:
            result = spatial.adaptive_threshold(image, block, constant)
            label = f"Adaptive (block {block}, C {constant})"
        elif use_otsu:
            result, otsu_value = spatial.otsu_threshold(image)
            label = f"Otsu (T = {otsu_value:.0f})"
            with controls:
                st.success(f"Otsu selected T = {otsu_value:.0f}")
        else:
            result = spatial.threshold_image(image, threshold, invert=invert)
            label = f"Binary (T = {threshold})"

        with preview:
            disp.compare(image, result, "Original", label)
        disp.download_image(result, download_name("threshold"), "dl_thr")

    # ---- Contrast stretching ----
    with tabs[3]:
        ui.concept(
            "Contrast Stretching",
            "A linear point transform that re-maps the darkest pixel to 0 and the "
            "brightest to 255, expanding a narrow histogram to fill the full range. "
            "Percentile limits are used so that a handful of outlier pixels cannot "
            "waste the whole dynamic range.",
            formula="output = (input - low) x 255 / (high - low)",
        )
        controls, preview = st.columns([1, 2.4])
        with controls:
            low_p = st.slider("Lower percentile", 0.0, 20.0, 2.0, 0.5, key="cs_low")
            high_p = st.slider("Upper percentile", 80.0, 100.0, 98.0, 0.5, key="cs_high")
            equalize = st.checkbox("Also show histogram equalisation", key="cs_eq")
            ui.reset_button(["cs_low", "cs_high", "cs_eq"], key="cs_reset")

        if low_p >= high_p:
            ui.warn("The lower percentile must be smaller than the upper percentile.")
            low_p, high_p = 2.0, 98.0

        result = spatial.contrast_stretch(image, low_p, high_p)
        with preview:
            disp.compare(image, result, "Original", "Contrast Enhanced")

        disp.histogram_pair(image, result, "Original Histogram", "Stretched Histogram")

        if equalize:
            ui.divider()
            equalized = spatial.histogram_equalization(image)
            disp.compare(result, equalized, "Contrast Stretched",
                         "Histogram Equalised")
            disp.download_image(equalized, download_name("equalized"), "dl_eq")

        disp.download_image(result, download_name("contrast_stretch"), "dl_cs")

    # ---- Histogram ----
    with tabs[4]:
        ui.concept(
            "Image Histogram",
            "The histogram counts how many pixels hold each intensity value. Its "
            "shape tells you at a glance whether an image is under-exposed "
            "(mass on the left), over-exposed (mass on the right) or low in "
            "contrast (a single narrow peak).",
        )
        operation = st.selectbox(
            "Compare the original histogram against",
            ["Negative", "Contrast Stretched", "Histogram Equalised",
             "Gamma 0.5", "Gamma 2.0"],
            key="hist_op",
        )
        mapping = {
            "Negative": lambda img: spatial.negative(img),
            "Contrast Stretched": lambda img: spatial.contrast_stretch(img),
            "Histogram Equalised": lambda img: spatial.histogram_equalization(img),
            "Gamma 0.5": lambda img: enhancement.gamma_correction(img, 0.5),
            "Gamma 2.0": lambda img: enhancement.gamma_correction(img, 2.0),
        }
        processed = mapping[operation](image)
        disp.compare(image, processed, "Original", operation)
        disp.histogram_pair(image, processed, "Original Histogram",
                            f"{operation} Histogram")
        if image.ndim == 3:
            disp.show_figure(disp.rgb_histogram_figure(image))
        disp.download_image(processed, download_name("histogram_op"), "dl_hist")

    # ---- Spatial filters ----
    with tabs[5]:
        _spatial_filters_tab(image)


def _spatial_filters_tab(image):
    """Low-pass / high-pass filtering (: spatial filtering)."""
    st.markdown("#### Spatial Filters")
    ui.note(
        "A spatial filter slides a small kernel over the image and replaces each "
        "pixel by a weighted sum of its neighbourhood. The kernel weights alone "
        "decide whether the filter smooths or sharpens."
    )

    choice = st.radio(
        "Filter type",
        ["Low-Pass (Average)", "Low-Pass (Gaussian)", "High-Pass",
         "High-Pass by Subtraction", "Sharpening", "Emboss"],
        horizontal=True,
        key="sf_choice",
    )

    controls, preview = st.columns([1, 2.4])

    if choice == "Low-Pass (Average)":
        ui.concept(
            "Low-Pass Filter (Average)",
            "Low-pass filters smooth an image and suppress high-frequency "
            "details and noise. Every neighbour carries the same weight, so the "
            "kernel is just a normalised box.",
            kernels=[("Average 3x3 (x 1/9)", np.ones((3, 3)))],
        )
        with controls:
            size = st.slider("Kernel size", 3, 25, 5, step=2, key="sf_avg_k")
            ui.reset_button(["sf_avg_k"], key="sf_avg_reset")
        result = filters.low_pass_average(image, size)
        label = f"Average Filter {size}x{size}"

    elif choice == "Low-Pass (Gaussian)":
        ui.concept(
            "Low-Pass Filter (Gaussian)",
            "Weights fall off with distance from the centre following a Gaussian, "
            "so the frequency roll-off is gradual and no ringing artefacts appear. "
            "Sigma controls how wide the bell is: larger sigma, stronger blur.",
            kernels=[("Gaussian 3x3 (x 1/16)", np.array([[1, 2, 1],
                                                         [2, 4, 2],
                                                         [1, 2, 1]]))],
        )
        with controls:
            size = st.slider("Kernel size", 3, 25, 5, step=2, key="sf_gauss_k")
            sigma = st.slider("Sigma", 0.0, 10.0, 1.5, 0.1, key="sf_gauss_s")
            ui.reset_button(["sf_gauss_k", "sf_gauss_s"], key="sf_gauss_reset")
        result = filters.low_pass_gaussian(image, size, sigma)
        label = f"Gaussian Filter {size}x{size}, sigma={sigma}"

    elif choice == "High-Pass":
        ui.concept(
            "High-Pass Filter",
            "High-pass filtering emphasises rapid intensity changes and fine "
            "details. The kernel coefficients sum to zero, so a perfectly flat "
            "region produces an output of zero and only transitions survive.",
            kernels=[("High-Pass (strong)", filters.HIGH_PASS_KERNEL),
                     ("High-Pass (soft)", filters.HIGH_PASS_KERNEL_SOFT)],
        )
        with controls:
            strong = st.checkbox("Use the strong 8-neighbour kernel", True,
                                 key="sf_hp_strong")
            ui.reset_button(["sf_hp_strong"], key="sf_hp_reset")
        result = filters.high_pass_filter(image, strong)
        label = "High-Pass Filter"

    elif choice == "High-Pass by Subtraction":
        ui.concept(
            "High-Pass by Subtraction",
            "A high-pass filter is exactly the complement of a low-pass one. "
            "Blur the image to keep only the low frequencies, then subtract that "
            "from the original and what remains is the high-frequency detail.",
            formula="high_pass = original - low_pass(original)",
        )
        with controls:
            size = st.slider("Low-pass kernel size", 3, 41, 9, step=2, key="sf_hps_k")
            ui.reset_button(["sf_hps_k"], key="sf_hps_reset")
        result = filters.high_pass_by_subtraction(image, size)
        label = "High-Pass (original - blurred)"

    elif choice == "Sharpening":
        ui.concept(
            "Sharpening",
            "Sharpening is a practical use of high-frequency enhancement: add the "
            "high-pass detail back on top of the original so edges gain local "
            "contrast. The kernel coefficients sum to 1, so average brightness "
            "is unchanged.",
            kernels=[("Sharpen 3x3", filters.SHARPEN_KERNEL)],
        )
        result = filters.sharpen_filter(image)
        label = "Sharpened"

    else:  # Emboss
        ui.concept(
            "Emboss",
            "A directional high-pass kernel. Edges facing one diagonal turn "
            "bright and the opposite ones turn dark, which reads as a 3-D relief.",
            kernels=[("Emboss 3x3", filters.EMBOSS_KERNEL)],
        )
        result = filters.emboss_filter(image)
        label = "Emboss"

    with preview:
        disp.compare(image, result, "Original", label)

    disp.before_after_slider(image, result, key="sf_wipe")
    disp.download_image(result, download_name("spatial_filter"), "dl_sf")


# --------------------------------------------------------------------------

def page_enhancement():
    ui.section_header(
        "Image Enhancement",
        "Enhancement adjusts intensities so that features of interest become "
        "easier to see. There is no single 'correct' result - only one that "
        "reveals what you are looking for.",
    )

    if not has_image():
        ui.no_image_notice()
        return

    image = get_image()
    tabs = st.tabs([
        "Brightness & Contrast", "Gamma Correction", "Sharpening",
        "Laplacian Sharpening", "Adaptive (CLAHE)",
    ])

    with tabs[0]:
        ui.concept(
            "Brightness and Contrast",
            "A linear point transform. Alpha is a gain that stretches intensities "
            "away from zero (contrast), beta is a bias that shifts them all up or "
            "down (brightness). Results are saturated at 0 and 255 so values "
            "never wrap around.",
            formula="output = alpha x input + beta\n"
                    "alpha = contrast (0.5 - 3.0)\n"
                    "beta  = brightness (-100 - +100)",
        )
        controls, preview = st.columns([1, 2.4])
        with controls:
            beta = st.slider("Brightness (beta)", -100, 100, 0, key="en_beta")
            alpha = st.slider("Contrast (alpha)", 0.5, 3.0, 1.0, 0.05, key="en_alpha")
            ui.reset_button(["en_beta", "en_alpha"], key="en_bc_reset")
        result = enhancement.brightness_contrast(image, alpha, beta)
        with preview:
            disp.compare(image, result, "Original",
                         f"alpha={alpha:.2f}, beta={beta}")
        disp.histogram_pair(image, result, "Original Histogram",
                            "Enhanced Histogram")
        disp.download_image(result, download_name("brightness_contrast"), "dl_bc")

    with tabs[1]:
        ui.concept(
            "Gamma Correction",
            "Gamma correction adjusts image intensity non-linearly. Because human "
            "brightness perception is itself non-linear, a power law recovers "
            "mid-tone detail that a linear brightness slider simply clips away. "
            "Gamma below 1 darkens, above 1 brightens.",
            formula="output = 255 x (input / 255) ^ (1 / gamma)",
        )
        controls, preview = st.columns([1, 2.4])
        with controls:
            gamma = st.slider("Gamma", 0.1, 3.0, 1.0, 0.05, key="en_gamma")
            ui.reset_button(["en_gamma"], key="en_gamma_reset")
        result = enhancement.gamma_correction(image, gamma)
        with preview:
            disp.compare(image, result, "Original", f"Gamma = {gamma:.2f}")
        disp.histogram_pair(image, result, "Original Histogram",
                            "Gamma Corrected Histogram")
        disp.download_image(result, download_name("gamma"), "dl_gamma")

    with tabs[2]:
        ui.concept(
            "Sharpening Kernel",
            "The centre weight of 5 keeps the original pixel while the four "
            "negative neighbours subtract the local average, which is the same as "
            "adding the high-pass detail back to the image. Edges therefore gain "
            "local contrast and look crisper.",
            kernels=[("Sharpen 3x3", enhancement.SHARPEN_KERNEL)],
        )
        result = enhancement.sharpen(image)
        disp.compare(image, result, "Original", "Sharpened")

        ui.divider()
        st.markdown("##### Unsharp Masking (adjustable)")
        controls, preview = st.columns([1, 2.4])
        with controls:
            size = st.slider("Blur kernel", 3, 21, 5, step=2, key="en_um_k")
            sigma = st.slider("Sigma", 0.1, 5.0, 1.0, 0.1, key="en_um_s")
            amount = st.slider("Amount", 0.0, 3.0, 1.0, 0.1, key="en_um_a")
            ui.reset_button(["en_um_k", "en_um_s", "en_um_a"], key="en_um_reset")
        unsharp = enhancement.unsharp_mask(image, size, sigma, amount)
        with preview:
            disp.compare(image, unsharp, "Original",
                         f"Unsharp Mask (amount {amount:.1f})")
        disp.before_after_slider(image, unsharp, key="en_um_wipe")
        disp.download_image(unsharp, download_name("sharpened"), "dl_sharp")

    with tabs[3]:
        ui.concept(
            "Laplacian Sharpening",
            "The Laplacian is a second-derivative operator: it swings negative on "
            "the bright side of an edge and positive on the dark side. Subtracting "
            "it from the original therefore exaggerates every intensity transition.",
            kernels=[("Laplacian (4-neighbour)", edge_detection.LAPLACIAN_4),
                     ("Laplacian (8-neighbour)", edge_detection.LAPLACIAN_8)],
            formula="sharpened = original - strength x Laplacian(original)",
        )
        controls, preview = st.columns([1, 2.4])
        with controls:
            strength = st.slider("Strength", 0.0, 3.0, 1.0, 0.1, key="en_lap_s")
            ui.reset_button(["en_lap_s"], key="en_lap_reset")
        result = enhancement.laplacian_sharpen(image, strength)
        with preview:
            disp.compare(image, result, "Original",
                         f"Laplacian Sharpened (x{strength:.1f})")
        disp.before_after_slider(image, result, key="en_lap_wipe")
        disp.download_image(result, download_name("laplacian_sharpened"), "dl_lapsharp")

    with tabs[4]:
        ui.concept(
            "CLAHE",
            "Contrast Limited Adaptive Histogram Equalisation equalises small "
            "tiles independently instead of the whole image, and clips tall "
            "histogram bins so that noise in flat regions is not amplified.",
        )
        controls, preview = st.columns([1, 2.4])
        with controls:
            clip = st.slider("Clip limit", 1.0, 10.0, 2.0, 0.5, key="en_clahe_c")
            tile = st.slider("Tile grid size", 2, 16, 8, key="en_clahe_t")
            ui.reset_button(["en_clahe_c", "en_clahe_t"], key="en_clahe_reset")
        result = enhancement.clahe(image, clip, tile)
        with preview:
            disp.compare(image, result, "Original", "CLAHE")
        disp.histogram_pair(image, result, "Original Histogram", "CLAHE Histogram")
        disp.download_image(result, download_name("clahe"), "dl_clahe")


# --------------------------------------------------------------------------

def page_noise():
    ui.section_header(
        "Noise Reduction & Smoothing",
        "Smoothing filters replace each pixel with a statistic of its "
        "neighbourhood. Which statistic you pick decides which noise you "
        "suppress and how much edge detail you lose.",
    )

    if not has_image():
        ui.no_image_notice()
        return

    image = get_image()

    with st.expander("Add synthetic noise (to test the filters)", expanded=False):
        noise_type = st.radio(
            "Noise model", ["None", "Salt & Pepper", "Gaussian"],
            horizontal=True, key="nz_type",
        )
        if noise_type == "Salt & Pepper":
            amount = st.slider("Noise amount", 0.0, 0.30, 0.05, 0.01, key="nz_sp")
            working = noise_removal.add_salt_pepper_noise(image, amount)
        elif noise_type == "Gaussian":
            sigma = st.slider("Noise sigma", 1.0, 80.0, 20.0, 1.0, key="nz_g")
            working = noise_removal.add_gaussian_noise(image, 0.0, sigma)
        else:
            working = image
        if noise_type != "None":
            st.caption(
                "Filters below now run on the noisy image. The clean original is "
                "still used as the reference for the PSNR figure."
            )

    tabs = st.tabs([
        "Gaussian Blur", "Median Filter", "Average Filter", "Bilateral Filter",
        "Compare All",
    ])

    with tabs[0]:
        ui.concept(
            "Gaussian Blur",
            "A weighted average where nearby pixels count more than distant ones. "
            "It is the natural choice for Gaussian (sensor) noise and preserves "
            "edges slightly better than a plain box filter.",
            kernels=[("Gaussian 3x3 (x 1/16)", np.array([[1, 2, 1],
                                                         [2, 4, 2],
                                                         [1, 2, 1]]))],
        )
        controls, preview = st.columns([1, 2.4])
        with controls:
            size = st.slider("Kernel size", 3, 31, 5, step=2, key="nr_g_k")
            sigma = st.slider("Sigma (0 = auto)", 0.0, 15.0, 1.5, 0.1, key="nr_g_s")
            ui.reset_button(["nr_g_k", "nr_g_s"], key="nr_g_reset")
        result = noise_removal.gaussian_blur(working, size, sigma)
        with preview:
            disp.compare(working, result, "Image", f"Gaussian {size}x{size}")
        _psnr_row(image, result)
        disp.download_image(result, download_name("gaussian_blur"), "dl_gb")

    with tabs[1]:
        ui.concept(
            "Median Filter",
            "A non-linear order-statistic filter: each pixel becomes the median of "
            "its neighbourhood. Median filtering is particularly useful for "
            "reducing salt-and-pepper noise, because an extreme outlier can never "
            "drag the median, and edges stay sharp.",
            formula="output(x, y) = median { neighbourhood of (x, y) }",
        )
        controls, preview = st.columns([1, 2.4])
        with controls:
            size = st.slider("Kernel size", 3, 21, 3, step=2, key="nr_m_k")
            ui.reset_button(["nr_m_k"], key="nr_m_reset")
        result = noise_removal.median_filter(working, size)
        with preview:
            disp.compare(working, result, "Image", f"Median {size}x{size}")
        _psnr_row(image, result)
        disp.download_image(result, download_name("median"), "dl_med")

    with tabs[2]:
        ui.concept(
            "Average / Mean Filter",
            "The simplest low-pass filter: every neighbour is weighted equally by "
            "1/(k x k). Cheap and effective against Gaussian noise, but it blurs "
            "edges just as strongly as it blurs the noise.",
            kernels=[("Average 3x3 (x 1/9)", np.ones((3, 3)))],
        )
        controls, preview = st.columns([1, 2.4])
        with controls:
            size = st.slider("Kernel size", 3, 31, 3, step=2, key="nr_a_k")
            ui.reset_button(["nr_a_k"], key="nr_a_reset")
        result = noise_removal.average_filter(working, size)
        with preview:
            disp.compare(working, result, "Image", f"Average {size}x{size}")
        _psnr_row(image, result)
        disp.download_image(result, download_name("average"), "dl_avg")

    with tabs[3]:
        ui.concept(
            "Bilateral Filter",
            "Edge-preserving smoothing. Neighbours are weighted by spatial "
            "distance AND by intensity difference, so pixels on the far side of an "
            "edge barely contribute. Flat regions get smoothed while edges stay "
            "crisp - at a noticeably higher computational cost.",
        )
        controls, preview = st.columns([1, 2.4])
        with controls:
            diameter = st.slider("Diameter", 1, 25, 9, key="nr_b_d")
            sigma_color = st.slider("Sigma Colour", 1, 200, 75, key="nr_b_c")
            sigma_space = st.slider("Sigma Space", 1, 200, 75, key="nr_b_s")
            ui.reset_button(["nr_b_d", "nr_b_c", "nr_b_s"], key="nr_b_reset")
        result = noise_removal.bilateral_filter(working, diameter,
                                                sigma_color, sigma_space)
        with preview:
            disp.compare(working, result, "Image", "Bilateral Filtered")
        _psnr_row(image, result)
        disp.download_image(result, download_name("bilateral"), "dl_bil")

    with tabs[4]:
        ui.concept(
            "Filter Comparison",
            "The same kernel size applied by four different filters. Watch how "
            "the median filter removes impulse noise without softening edges, "
            "while the mean and Gaussian filters trade sharpness for smoothness.",
        )
        size = st.slider("Kernel size for all filters", 3, 15, 5, step=2,
                         key="nr_cmp_k")
        results = {
            "Input": working,
            f"Gaussian {size}x{size}": noise_removal.gaussian_blur(working, size, 0),
            f"Median {size}x{size}": noise_removal.median_filter(working, size),
            f"Average {size}x{size}": noise_removal.average_filter(working, size),
            "Bilateral (9, 75, 75)": noise_removal.bilateral_filter(working, 9, 75, 75),
        }
        disp.grid(results, columns=3)

        rows = []
        for name, output in results.items():
            metrics = noise_removal.denoise_metrics(image, output)
            if metrics:
                psnr = metrics["psnr"]
                rows.append({
                    "Filter": name,
                    "MSE": round(metrics["mse"], 2),
                    # Kept as text so the column stays a single type even when
                    # an unfiltered image gives an infinite PSNR.
                    "PSNR (dB)": "inf" if np.isinf(psnr) else f"{psnr:.2f}",
                })
        if rows:
            st.markdown("##### Quality against the clean original")
            ui.data_table(rows)


def _psnr_row(reference, processed):
    """Show MSE / PSNR of a filtered result against the clean original."""
    metrics = noise_removal.denoise_metrics(reference, processed)
    if not metrics:
        return
    psnr = "inf" if np.isinf(metrics["psnr"]) else f"{metrics['psnr']:.2f}"
    ui.stat_row([
        ("MSE vs Original", round(metrics["mse"], 2)),
        ("PSNR vs Original", psnr, "dB"),
    ])


# --------------------------------------------------------------------------

def page_gradients():
    ui.section_header(
        "Edge Detection",
        "The image gradient measures how fast intensity changes. Each operator "
        "approximates the partial derivatives with a small convolution mask; "
        "edges are where the gradient magnitude is large.",
    )

    if not has_image():
        ui.no_image_notice()
        return

    image = get_image()
    ui.formula_block(
        "magnitude = sqrt(Gx^2 + Gy^2)\n"
        "direction = arctan2(Gy, Gx)",
        name="Gradient magnitude and direction",
    )

    tabs = st.tabs([
        "Roberts", "Prewitt", "Sobel", "Magnitude & Direction", "Compare All",
    ])

    with tabs[0]:
        ui.concept(
            "Roberts Cross-Gradient",
            "The smallest possible gradient operator: two 2x2 masks that take "
            "differences along the diagonals. It is fast and gives very thin edge "
            "responses, but with only four pixels in the neighbourhood it is "
            "extremely sensitive to noise.",
            kernels=[("Roberts Gx", gradients.ROBERTS_X),
                     ("Roberts Gy", gradients.ROBERTS_Y)],
        )
        result = gradients.roberts_operator(image)
        _gradient_view(image, result, "roberts")

    with tabs[1]:
        ui.concept(
            "Prewitt Operator",
            "A 3x3 mask that differences one direction while averaging uniformly "
            "along the perpendicular direction. The extra averaging gives it "
            "noticeably better noise immunity than Roberts.",
            kernels=[("Prewitt Gx", gradients.PREWITT_X),
                     ("Prewitt Gy", gradients.PREWITT_Y)],
        )
        result = gradients.prewitt_operator(image)
        _gradient_view(image, result, "prewitt")

    with tabs[2]:
        ui.concept(
            "Sobel Operator",
            "The Sobel operator estimates the image gradient in the horizontal and "
            "vertical directions and is commonly used for edge detection. It is "
            "Prewitt with the centre row and column weighted 2, which adds a mild "
            "Gaussian smoothing and therefore better noise rejection.",
            kernels=[("Sobel Gx", gradients.SOBEL_X),
                     ("Sobel Gy", gradients.SOBEL_Y)],
        )
        kernel_size = st.select_slider("Sobel aperture size", [1, 3, 5, 7], value=3,
                                       key="gr_sobel_k")
        result = gradients.sobel_operator(image, kernel_size)
        _gradient_view(image, result, "sobel")

    with tabs[3]:
        ui.concept(
            "Gradient Magnitude & Direction",
            "Magnitude answers 'how strong is the edge here', direction answers "
            "'which way does intensity climb'. The direction image below encodes "
            "the angle as hue and the magnitude as brightness, so flat areas stay "
            "black and each edge orientation gets its own colour.",
            formula="magnitude = sqrt(Gx^2 + Gy^2)\n"
                    "direction = arctan2(Gy, Gx)   [-180, +180 degrees]",
        )
        operator = st.selectbox("Operator", ["Sobel", "Prewitt", "Roberts", "Scharr"],
                                key="gr_md_op")
        engine = {
            "Sobel": lambda img: gradients.sobel_operator(img, 3),
            "Prewitt": gradients.prewitt_operator,
            "Roberts": gradients.roberts_operator,
            "Scharr": gradients.scharr_operator,
        }[operator]
        result = engine(image)
        direction_rgb = gradients.direction_visualization(
            result["magnitude_raw"], result["direction_raw"]
        )
        disp.grid({
            "Original": image,
            f"{operator} Magnitude": result["magnitude"],
            "Gradient Direction (hue = angle)": direction_rgb,
        }, columns=3)

        angles = result["direction_raw"]
        magnitude = result["magnitude_raw"]
        strong = angles[magnitude > np.percentile(magnitude, 90)]
        ui.stat_row([
            ("Max Magnitude", round(float(magnitude.max()), 1)),
            ("Mean Magnitude", round(float(magnitude.mean()), 2)),
            ("Dominant Angle", f"{float(np.median(strong)):.0f}" if strong.size else "-",
             "deg"),
        ])
        disp.download_image(result["magnitude"], download_name("gradient_magnitude"),
                            "dl_gmag")

    with tabs[4]:
        ui.concept(
            "Operator Comparison",
            "The same image through all three classical operators. Roberts gives "
            "the thinnest but noisiest response, Prewitt is smoother, and Sobel "
            "usually gives the cleanest edges of the three.",
        )
        results = gradients.all_operators(image)
        disp.grid({
            "Original": image,
            "Roberts Magnitude": results["Roberts"]["magnitude"],
            "Prewitt Magnitude": results["Prewitt"]["magnitude"],
            "Sobel Magnitude": results["Sobel"]["magnitude"],
        }, columns=2)


def _gradient_view(image, result, key):
    """Shared X / Y / magnitude layout for a gradient operator."""
    disp.grid({
        "Original": image,
        "Gx (horizontal derivative)": result["x"],
        "Gy (vertical derivative)": result["y"],
        "Gradient Magnitude": result["magnitude"],
    }, columns=2)

    choice = st.selectbox(
        "Download which result?", ["Magnitude", "Gx", "Gy"], key=f"gr_{key}_dl",
    )
    mapping = {"Magnitude": result["magnitude"], "Gx": result["x"], "Gy": result["y"]}
    disp.download_image(mapping[choice], download_name(f"{key}_{choice.lower()}"),
                        f"dl_{key}")


# --------------------------------------------------------------------------

def page_edges():
    ui.section_header(
        "Edge Detection",
        "First-derivative operators mark an edge where the gradient is a "
        "maximum. Second-derivative operators mark an edge where the second "
        "derivative crosses zero.",
    )

    if not has_image():
        ui.no_image_notice()
        return

    image = get_image()

    controls = st.columns([1, 1, 1, 1])
    with controls[0]:
        sobel_k = st.select_slider("Sobel aperture", [1, 3, 5, 7], value=3,
                                   key="ed_sobel_k")
    with controls[1]:
        log_sigma = st.slider("LoG sigma", 0.4, 5.0, 1.4, 0.1, key="ed_log_sigma")
    with controls[2]:
        binarize = st.checkbox("Binarise edge maps", key="ed_binary")
    with controls[3]:
        threshold = st.slider("Binary threshold", 0, 255, 40, key="ed_thr",
                              disabled=not binarize)

    ui.reset_button(["ed_sobel_k", "ed_log_sigma", "ed_binary", "ed_thr"],
                    key="ed_reset")
    ui.divider()

    applied_threshold = threshold if binarize else None
    results = edge_detection.compare_edge_detectors(
        image, sobel_k, log_sigma, applied_threshold
    )

    st.markdown("#### Comparison")
    disp.grid({"Original": image, **results}, columns=3)

    ui.divider()
    st.markdown("#### Operator details")

    detail_tabs = st.tabs([
        "Roberts", "Prewitt", "Sobel", "Laplacian", "Laplacian of Gaussian",
    ])

    with detail_tabs[0]:
        ui.concept(
            "Roberts",
            "Two 2x2 diagonal difference masks. Cheapest first-derivative edge "
            "detector; thin edges, but very noise-sensitive.",
            kernels=[("Roberts Gx", gradients.ROBERTS_X),
                     ("Roberts Gy", gradients.ROBERTS_Y)],
            expanded=True,
        )
        disp.compare(image, results["Roberts"], "Original", "Roberts")
        disp.download_image(results["Roberts"], download_name("roberts"), "dl_ed_rob")

    with detail_tabs[1]:
        ui.concept(
            "Prewitt",
            "3x3 masks that difference one axis and average uniformly along the "
            "other, giving better noise tolerance than Roberts.",
            kernels=[("Prewitt Gx", gradients.PREWITT_X),
                     ("Prewitt Gy", gradients.PREWITT_Y)],
            expanded=True,
        )
        disp.compare(image, results["Prewitt"], "Original", "Prewitt")
        disp.download_image(results["Prewitt"], download_name("prewitt"), "dl_ed_pre")

    with detail_tabs[2]:
        ui.concept(
            "Sobel",
            "Prewitt with a centre weight of 2, which adds mild smoothing. The "
            "most widely used first-derivative edge operator.",
            kernels=[("Sobel Gx", gradients.SOBEL_X),
                     ("Sobel Gy", gradients.SOBEL_Y)],
            expanded=True,
        )
        disp.compare(image, results["Sobel"], "Original", "Sobel")
        disp.download_image(results["Sobel"], download_name("sobel"), "dl_ed_sob")

    with detail_tabs[3]:
        ui.concept(
            "Laplacian",
            "A second-derivative operator, L = d2I/dx2 + d2I/dy2. It is isotropic "
            "(it has no preferred edge direction) but, being a second derivative, "
            "it amplifies noise heavily.",
            kernels=[("Laplacian (4-neighbour)", edge_detection.LAPLACIAN_4),
                     ("Laplacian (8-neighbour)", edge_detection.LAPLACIAN_8)],
            expanded=True,
        )
        disp.compare(image, results["Laplacian"], "Original", "Laplacian")
        disp.download_image(results["Laplacian"], download_name("laplacian"),
                            "dl_ed_lap")

    with detail_tabs[4]:
        ui.concept(
            "Laplacian of Gaussian (Marr-Hildreth)",
            "Smooth with a Gaussian first, then apply the Laplacian. The Gaussian "
            "controls which scale of detail survives, and the true edges are the "
            "zero-crossings of the result.",
            formula="LoG = Laplacian( Gaussian(image, sigma) )\n"
                    "edges = zero-crossings of LoG",
            expanded=True,
        )
        disp.compare(image, results["Laplacian of Gaussian"], "Original", "LoG")
        crossings = edge_detection.zero_crossing(image, 5, log_sigma)
        disp.compare(results["Laplacian of Gaussian"], crossings,
                     "LoG Response", "Zero Crossings")
        disp.download_image(crossings, download_name("log_zero_crossings"),
                            "dl_ed_log")


# --------------------------------------------------------------------------

def page_canny():
    ui.section_header(
        "Canny",
        "Detect image boundaries with adjustable thresholds and smoothing controls.",
    )

    if not has_image():
        ui.no_image_notice()
        return

    image = get_image()

    tabs = st.tabs([
        "Detector", "Compare Thresholds",
    ])

    # ---- Interactive detector ----
    with tabs[0]:
        controls, preview = st.columns([1, 2.4])

        with controls:
            preset = st.selectbox(
                "Preset", ["Custom"] + list(canny_ops.PRESETS.keys()),
                key="cn_preset",
            )
            if preset != "Custom":
                preset_low, preset_high = canny_ops.PRESETS[preset]
                if st.button(f"Apply preset ({preset_low} / {preset_high})",
                             key="cn_apply_preset", width="stretch"):
                    st.session_state["cn_low"] = preset_low
                    st.session_state["cn_high"] = preset_high
                    st.rerun()

            blur_kernel = st.select_slider(
                "Gaussian blur kernel", [1, 3, 5, 7, 9, 11], value=5, key="cn_blur",
            )
            low = st.slider("Lower threshold", 0, 255, 100, key="cn_low")
            high = st.slider("Upper threshold", 0, 255, 200, key="cn_high")
            aperture = st.select_slider("Aperture size (Sobel)", [3, 5, 7], value=3,
                                        key="cn_aperture")
            l2 = st.checkbox("Use L2 gradient (exact magnitude)", True, key="cn_l2")
            ui.reset_button(
                ["cn_blur", "cn_low", "cn_high", "cn_aperture", "cn_l2", "cn_preset"],
                key="cn_reset",
            )

        ui.validate_thresholds(low, high)

        result = canny_ops.canny_edge_detection(image, low, high, blur_kernel,
                                                aperture, l2)
        with preview:
            disp.compare(image, result, "Original",
                         f"Canny ({min(low, high)} / {max(low, high)})")

        ratio = canny_ops.edge_pixel_ratio(result)
        ui.stat_row([
            ("Edge Pixels", f"{int(np.count_nonzero(result)):,}"),
            ("Edge Density", round(ratio, 2), "%"),
            ("Threshold Ratio", f"1 : {max(low, high) / max(min(low, high), 1):.1f}"),
            ("Blur Kernel", f"{blur_kernel}x{blur_kernel}"),
        ])

        st.write("")
        disp.before_after_slider(image, result, key="cn_wipe")
        disp.download_image(result, download_name("canny"), "dl_canny")

    # ---- Preset comparison ----
    with tabs[1]:
        ui.concept(
            "Compare Thresholds",
            "The two thresholds decide which gradient ridges become edges. Lower "
            "them and you catch faint detail along with noise; raise them and only "
            "the strongest boundaries survive. Canny recommended a high:low ratio "
            "between 2:1 and 3:1.",
        )
        compare_blur = st.select_slider("Gaussian blur kernel", [1, 3, 5, 7, 9],
                                        value=5, key="cn_cmp_blur")
        preset_results = canny_ops.compare_presets(image, compare_blur, 3)
        disp.grid({"Original": image, **preset_results}, columns=2)

        rows = []
        for name, (low_value, high_value) in canny_ops.PRESETS.items():
            edges = canny_ops.canny_edge_detection(image, low_value, high_value,
                                                   compare_blur, 3)
            rows.append({
                "Preset": name,
                "Low": low_value,
                "High": high_value,
                "Edge Pixels": int(np.count_nonzero(edges)),
                "Edge Density (%)": round(canny_ops.edge_pixel_ratio(edges), 2),
            })
        ui.data_table(rows)

        ui.divider()
        st.markdown("##### Manual side-by-side")
        left, right = st.columns(2)
        with left:
            low_a = st.slider("A - low", 0, 255, 50, key="cn_a_low")
            high_a = st.slider("A - high", 0, 255, 100, key="cn_a_high")
        with right:
            low_b = st.slider("B - low", 0, 255, 150, key="cn_b_low")
            high_b = st.slider("B - high", 0, 255, 250, key="cn_b_high")

        if low_a > high_a or low_b > high_b:
            ui.warn(
                "A lower threshold exceeds its upper threshold. The values are "
                "swapped automatically so the comparison still renders."
            )

        edges_a = canny_ops.canny_edge_detection(image, low_a, high_a, compare_blur, 3)
        edges_b = canny_ops.canny_edge_detection(image, low_b, high_b, compare_blur, 3)
        disp.compare(edges_a, edges_b,
                     f"A ({min(low_a, high_a)} / {max(low_a, high_a)})",
                     f"B ({min(low_b, high_b)} / {max(low_b, high_b)})")



# --------------------------------------------------------------------------

def page_shapes():
    ui.section_header(
        "Shape Detection",
        "Classical, model-free shape recognition: grayscale, blur, Canny, "
        "contour extraction, then polygon approximation. The vertex count of "
        "the approximating polygon names the shape.",
    )

    if not has_image():
        ui.no_image_notice()
        return

    image = get_image()

    tabs = st.tabs(["Shape Detection", "Contours"])

    # ---- Shapes ----
    with tabs[0]:
        ui.concept(
            "Polygon Approximation",
            "approxPolyDP (Ramer-Douglas-Peucker) replaces a noisy contour with the "
            "smallest polygon that stays within epsilon of it. Epsilon is set as a "
            "fraction of the perimeter, so it scales with the object. Three "
            "vertices means a triangle, four a square or rectangle, five a "
            "pentagon; a contour with many vertices is called a circle when its "
            "circularity is close to 1.",
            formula="epsilon      = factor x perimeter\n"
                    "circularity  = 4 x pi x Area / Perimeter^2   (1.0 = perfect circle)",
        )

        controls, preview = st.columns([1, 2.4])
        with controls:
            blur_kernel = st.select_slider("Gaussian blur kernel", [1, 3, 5, 7, 9],
                                           value=5, key="sh_blur")
            low = st.slider("Canny lower threshold", 0, 255, 50, key="sh_low")
            high = st.slider("Canny upper threshold", 0, 255, 150, key="sh_high")
            min_area = st.slider("Minimum contour area (px)", 50, 20000, 500, 50,
                                 key="sh_area")
            epsilon = st.slider("Approximation factor (epsilon)", 0.01, 0.10, 0.04,
                                0.005, key="sh_eps")
            draw_boxes = st.checkbox("Draw bounding boxes", True, key="sh_boxes")
            draw_labels = st.checkbox("Draw shape labels", True, key="sh_labels")
            ui.reset_button(
                ["sh_blur", "sh_low", "sh_high", "sh_area", "sh_eps",
                 "sh_boxes", "sh_labels"],
                key="sh_reset",
            )

        ui.validate_thresholds(low, high)

        annotated, detections = shapes.detect_shapes(
            image, blur_kernel, low, high, min_area, epsilon, draw_boxes, draw_labels
        )

        with preview:
            disp.compare(image, annotated, "Original", "Detected Shapes")

        if not detections:
            ui.warn(
                "No shapes found. Try lowering the minimum contour area, or "
                "reducing the Canny thresholds so more edges are detected."
            )
        else:
            counts = {}
            for item in detections:
                counts[item["shape"]] = counts.get(item["shape"], 0) + 1
            ui.stat_row(
                [("Shapes Found", len(detections))]
                + [(name, count) for name, count in sorted(counts.items())][:4]
            )
            st.markdown("##### Detected shapes")
            ui.data_table([
                {
                    "#": index,
                    "Shape": item["shape"],
                    "Vertices": item["vertices"],
                    "Area (px)": round(item["area"], 1),
                    "Perimeter (px)": round(item["perimeter"], 1),
                    "Bounding Box (x, y, w, h)": ", ".join(
                        str(v) for v in item["bbox"]
                    ),
                }
                for index, item in enumerate(detections, start=1)
            ])

        disp.download_image(annotated, download_name("shapes"), "dl_shapes")

    # ---- Contours ----
    with tabs[1]:
        ui.concept(
            "Contours",
            "A contour is a curve joining continuous points along a boundary of "
            "equal intensity. RETR_EXTERNAL keeps only the outermost boundary of "
            "each object; RETR_LIST returns every contour, including holes inside "
            "shapes.",
        )

        controls, preview = st.columns([1, 2.4])
        with controls:
            mode = st.radio("Retrieval mode", ["External contours", "All contours"],
                            key="ct_mode")
            blur_kernel = st.select_slider("Gaussian blur kernel", [1, 3, 5, 7, 9],
                                           value=5, key="ct_blur")
            low = st.slider("Canny lower threshold", 0, 255, 50, key="ct_low")
            high = st.slider("Canny upper threshold", 0, 255, 150, key="ct_high")
            min_area = st.slider("Minimum contour area (px)", 10, 10000, 100, 10,
                                 key="ct_area")
            boxes = st.checkbox("Draw bounding rectangles", True, key="ct_boxes")
            ui.reset_button(
                ["ct_mode", "ct_blur", "ct_low", "ct_high", "ct_area", "ct_boxes"],
                key="ct_reset",
            )

        ui.validate_thresholds(low, high)

        retrieval = "external" if mode == "External contours" else "all"
        contours, edges = shapes.find_contours(image, retrieval, blur_kernel,
                                               low, high, min_area)
        drawn = shapes.draw_contours(image, contours, boxes)

        with preview:
            disp.compare(image, drawn, "Original",
                         f"{len(contours)} contour(s) - {mode}")

        ui.stat_row([
            ("Contour Count", len(contours)),
            ("Retrieval Mode", "RETR_EXTERNAL" if retrieval == "external"
             else "RETR_LIST"),
            ("Edge Pixels", f"{int(np.count_nonzero(edges)):,}"),
        ])

        if contours:
            st.markdown("##### Contour measurements")
            ui.data_table(shapes.contour_summary(contours))
        else:
            ui.warn("No contours passed the minimum-area filter.")

        with st.expander("Show the intermediate Canny edge map"):
            st.image(edges, width="stretch")

        disp.download_image(drawn, download_name("contours"), "dl_contours")


# ==========================================================================
# Entry point
# ==========================================================================

ROUTES = {
    "Home": page_home,
    "Image Processing": page_spatial,
    "Enhancement": page_enhancement,
    "Filters": page_noise,
    "Gradients": page_gradients,
    "Edge Detection": page_edges,
    "Canny": page_canny,
    "Shape Detection": page_shapes,
}


def main():
    st.set_page_config(
        page_title="Image Processing",
        page_icon="◧",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_state()
    ui_theme.apply_theme()

    render_sidebar()
    ui.page_header()

    page = st.session_state.get("page", "Home")
    try:
        ROUTES.get(page, page_home)()
    except Exception as error:  # keep the app alive on any unexpected failure
        st.error(f"Something went wrong while rendering this section: {error}")
        st.caption(
            "Try adjusting the parameters, or press Reset Image in the sidebar "
            "and load the image again."
        )


if __name__ == "__main__":
    main()
