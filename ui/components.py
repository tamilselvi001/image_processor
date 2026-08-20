"""Reusable UI pieces: headers, cards, notes, kernel blocks, reset buttons."""

import html

import numpy as np
import streamlit as st

APP_TITLE = "Image Processing Studio"
APP_SUBTITLE = "Explore, enhance and analyze images with classical computer vision techniques."
APP_DESCRIPTION = "Classical image processing and computer vision tools"


def page_header():
    """The fixed application header shown on every page."""
    st.markdown(
        f"""
        <div class="iva-header">
            <h1>{APP_TITLE}</h1>
            <div class="iva-sub">{APP_SUBTITLE}</div>
            <p class="iva-desc">{APP_DESCRIPTION}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title, description=None, tag=None):
    """Heading for a tool page, with an optional short label chip."""
    tag_html = f'<span class="iva-tag">{html.escape(tag)}</span>' if tag else ""
    desc_html = f"<p>{html.escape(description)}</p>" if description else ""
    st.markdown(
        f"""
        <div class="iva-section">
            <h2>{html.escape(title)}{tag_html}</h2>
            {desc_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def note(text):
    """Subtle informational strip."""
    st.markdown(f'<div class="iva-note">{text}</div>', unsafe_allow_html=True)


def warn(text):
    """Subtle warning strip (used for invalid parameter combinations)."""
    st.markdown(f'<div class="iva-warn">{text}</div>', unsafe_allow_html=True)


def stat_card(label, value, unit=""):
    """One metric tile."""
    unit_html = f'<span class="iva-unit">{html.escape(str(unit))}</span>' if unit else ""
    st.markdown(
        f"""
        <div class="iva-card">
            <div class="iva-label">{html.escape(str(label))}</div>
            <div class="iva-value">{html.escape(str(value))}{unit_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_row(pairs):
    """Render a row of metric tiles from [(label, value, unit), ...]."""
    columns = st.columns(len(pairs))
    for column, item in zip(columns, pairs):
        label, value = item[0], item[1]
        unit = item[2] if len(item) > 2 else ""
        with column:
            stat_card(label, value, unit)


def image_info_cards(info):
    """Width / height / channels / type tiles for the uploaded image."""
    stat_row([
        ("Width", info["width"], "px"),
        ("Height", info["height"], "px"),
        ("Channels", info["channels"]),
        ("Image Type", info["type"]),
        ("Data Type", info["dtype"]),
    ])


def kernel_block(name, matrix, width=5):
    """Render a convolution kernel as an aligned monospace grid."""
    array = np.asarray(matrix)
    rows = []
    for row in np.atleast_2d(array):
        rows.append("".join(f"{_fmt(v):>{width}}" for v in row))
    body = "\n".join(rows)
    st.markdown(
        f'<div class="iva-kernel-name">{html.escape(name)}</div>'
        f'<div class="iva-kernel">{html.escape(body)}</div>',
        unsafe_allow_html=True,
    )


def formula_block(text, name="Formula"):
    """Render a formula (or a small pipeline diagram) in a monospace block."""
    st.markdown(
        f'<div class="iva-kernel-name">{html.escape(name)}</div>'
        f'<div class="iva-kernel">{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def _fmt(value):
    value = float(value)
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.3g}"


def hint(text):
    """One-line description of what the current operation does."""
    st.markdown(
        f'<div class="iva-hint">{text}</div>',
        unsafe_allow_html=True,
    )


def concept(title, text, formula=None, kernels=None, expanded=False):
    """Render compact optional details for a processing operator.

    Kept intentionally lightweight: the operator description, optional formula,
    and optional kernels live in a collapsible panel so the main image/result
    area stays clean.
    """
    with st.expander(title, expanded=expanded):
        if text:
            st.markdown(f'<div class="iva-hint">{text}</div>', unsafe_allow_html=True)
        if formula:
            formula_block(formula)
        if kernels:
            columns = st.columns(min(len(kernels), 3))
            for index, (name, matrix) in enumerate(kernels):
                with columns[index % len(columns)]:
                    kernel_block(name, matrix)


def details(kernels=None, formula=None, label="Kernel / parameters"):
    """Optional collapsed panel holding kernels and formulas for reference.

    kernels : list of (name, matrix) tuples
    formula : string shown in a monospace block
    """
    if not kernels and not formula:
        return
    with st.expander(label, expanded=False):
        if formula:
            formula_block(formula)
        if kernels:
            columns = st.columns(min(len(kernels), 3))
            for index, (name, matrix) in enumerate(kernels):
                with columns[index % len(columns)]:
                    kernel_block(name, matrix)


def no_image_notice():
    """Empty state shown on a tool page when no image is loaded."""
    st.markdown(
        '<div class="iva-warn">No image loaded. Use the image picker in the sidebar to '
        "sidebar and add a JPG, JPEG, PNG or BMP file, or load one of the "
        "built-in samples.</div>",
        unsafe_allow_html=True,
    )


def reset_button(keys, label="Restore Original", key=None, help_text=None):
    """Clear the widget state listed in `keys` and rerun.

    The uploaded image itself is never touched - only the control values -
    because every processing function already works on a copy.
    """
    clicked = st.button(
        label,
        key=key,
        help=help_text or "Reset this section's controls to their defaults.",
        width="stretch",
    )
    if clicked:
        for name in keys:
            st.session_state.pop(name, None)
        st.rerun()
    return clicked


def validate_thresholds(low, high):
    """Return True when low <= high, otherwise show a warning strip."""
    if low > high:
        warn(
            f"Lower threshold ({low}) is above the upper threshold ({high}). "
            "The values were swapped so processing can continue."
        )
        return False
    return True


def data_table(rows, max_rows=50):
    """Render a list of dicts as a themed HTML table.

    Streamlit's own dataframe is drawn on a canvas, so it cannot follow the
    injected stylesheet. These result tables are small, so a plain HTML
    table gives a consistent look in both themes (and avoids Arrow type
    conversion entirely).
    """
    if not rows:
        return
    columns = list(rows[0].keys())
    visible = rows[:max_rows]

    head = "".join(f"<th>{html.escape(str(name))}</th>" for name in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(row.get(name, '')))}</td>" for name in columns)
        + "</tr>"
        for row in visible
    )
    st.markdown(
        f'<div class="iva-table-wrap"><table class="iva-table">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>",
        unsafe_allow_html=True,
    )
    if len(rows) > max_rows:
        st.caption(f"Showing the first {max_rows} of {len(rows)} rows.")


def divider(space_before="1.4rem", space_after="1.1rem"):
    """Thin separator with controlled spacing."""
    st.markdown(
        f'<div style="height:1px;background:var(--iva-border);'
        f'margin:{space_before} 0 {space_after} 0;"></div>',
        unsafe_allow_html=True,
    )
