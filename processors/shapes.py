"""Contour detection and classical shape recognition.

No machine learning is involved.  The pipeline is purely geometric:

    Grayscale -> Gaussian blur -> Canny -> findContours
              -> approxPolyDP  -> classify by vertex count

The number of vertices of the approximating polygon tells us the shape.
"""

import cv2
import numpy as np

from .utils import ensure_color, ensure_gray, odd_kernel

# Distinct, readable colours (RGB) for the drawn overlays.
SHAPE_COLORS = {
    "Triangle": (239, 108, 91),
    "Square": (86, 173, 121),
    "Rectangle": (91, 154, 239),
    "Pentagon": (206, 147, 216),
    "Hexagon": (255, 193, 94),
    "Circle": (77, 208, 225),
    "Polygon": (168, 168, 168),
}


def classify_shape(approx, contour):
    """Name a shape from its approximating polygon.

    3 vertices -> Triangle
    4 vertices -> Square if the aspect ratio is ~1, otherwise Rectangle
    5 / 6      -> Pentagon / Hexagon
    more       -> Circle if the contour is nearly circular, else Polygon

    Circularity = 4*pi*Area / Perimeter^2, which equals 1.0 for a perfect
    circle and drops as the outline becomes more irregular.
    """
    vertices = len(approx)

    if vertices == 3:
        return "Triangle"

    if vertices == 4:
        _, _, width, height = cv2.boundingRect(approx)
        aspect_ratio = width / float(height) if height else 0.0
        return "Square" if 0.92 <= aspect_ratio <= 1.08 else "Rectangle"

    if vertices == 5:
        return "Pentagon"

    if vertices == 6:
        return "Hexagon"

    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if perimeter <= 0:
        return "Polygon"
    circularity = 4.0 * np.pi * area / (perimeter ** 2)
    return "Circle" if circularity > 0.80 else "Polygon"


def detect_shapes(image, blur_kernel=5, low_threshold=50, high_threshold=150,
                  min_area=500, epsilon_factor=0.04, draw_boxes=True,
                  draw_labels=True):
    """Detect and label basic shapes.

    epsilon_factor controls how aggressively approxPolyDP simplifies the
    contour: epsilon = epsilon_factor * perimeter.  Too small keeps noisy
    vertices, too large collapses everything into a triangle.

    Returns (annotated_rgb_image, list_of_detections).
    """
    output = ensure_color(image).copy()
    gray = ensure_gray(image)

    blur_kernel = odd_kernel(blur_kernel, minimum=1)
    if blur_kernel > 1:
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    low, high = int(low_threshold), int(high_threshold)
    if low > high:
        low, high = high, low
    edges = cv2.Canny(gray, low, high)

    # Close 1-pixel gaps so broken outlines still form a closed contour.
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < float(min_area):
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(contour, float(epsilon_factor) * perimeter, True)
        name = classify_shape(approx, contour)
        color = SHAPE_COLORS.get(name, SHAPE_COLORS["Polygon"])

        cv2.drawContours(output, [approx], -1, color, 2)

        x, y, width, height = cv2.boundingRect(approx)
        if draw_boxes:
            cv2.rectangle(output, (x, y), (x + width, y + height), color, 1)

        if draw_labels:
            _put_label(output, name, x, y, color)

        detections.append({
            "shape": name,
            "vertices": int(len(approx)),
            "area": float(area),
            "perimeter": float(perimeter),
            "bbox": (int(x), int(y), int(width), int(height)),
        })

    detections.sort(key=lambda item: item["area"], reverse=True)
    return output, detections


def _put_label(canvas, text, x, y, color):
    """Draw a shape label with a filled background so it stays readable."""
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(text_h + baseline + 2, y)
    cv2.rectangle(canvas, (x, top - text_h - baseline - 2),
                  (x + text_w + 6, top), color, -1)
    cv2.putText(canvas, text, (x + 3, top - baseline),
                font, scale, (20, 20, 20), thickness, cv2.LINE_AA)


def find_contours(image, mode="external", blur_kernel=5,
                  low_threshold=50, high_threshold=150, min_area=100):
    """Find contours using either RETR_EXTERNAL (outermost only) or
    RETR_LIST (every contour, including holes inside shapes).
    """
    gray = ensure_gray(image)
    blur_kernel = odd_kernel(blur_kernel, minimum=1)
    if blur_kernel > 1:
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    low, high = int(low_threshold), int(high_threshold)
    if low > high:
        low, high = high, low
    edges = cv2.Canny(gray, low, high)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    retrieval = cv2.RETR_EXTERNAL if mode == "external" else cv2.RETR_LIST
    contours, _ = cv2.findContours(edges, retrieval, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= float(min_area)], edges


def draw_contours(image, contours, draw_boxes=True, thickness=2,
                  color=(91, 154, 239), box_color=(255, 193, 94)):
    """Overlay contours (and optionally their bounding rectangles)."""
    output = ensure_color(image).copy()
    cv2.drawContours(output, contours, -1, color, thickness)
    if draw_boxes:
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            cv2.rectangle(output, (x, y), (x + width, y + height), box_color, 1)
    return output


def contour_summary(contours):
    """Per-contour area / perimeter / bounding box, largest first."""
    rows = []
    for index, contour in enumerate(contours, start=1):
        x, y, width, height = cv2.boundingRect(contour)
        rows.append({
            "#": index,
            "Area (px)": round(float(cv2.contourArea(contour)), 1),
            "Perimeter (px)": round(float(cv2.arcLength(contour, True)), 1),
            "Bounding Box (x, y, w, h)": "%d, %d, %d, %d" % (x, y, width, height),
        })
    rows.sort(key=lambda row: row["Area (px)"], reverse=True)
    for position, row in enumerate(rows, start=1):
        row["#"] = position
    return rows
