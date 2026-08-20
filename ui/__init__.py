"""Presentation layer for Image Processing Studio.

    theme.py         dark / light palettes and the injected stylesheet
    components.py    headers, cards, notes, kernel blocks, reset buttons
    image_display.py comparisons, grids, histograms and downloads
"""

from . import components, image_display, theme  # noqa: F401

__all__ = ["theme", "components", "image_display"]
