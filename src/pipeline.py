"""Classical CV pipeline: 3D floor-plan render -> room polygons and areas.

Stages, in order:

1. Separate the apartment from the white page.
2. Keep long axis-aligned edges that sit on a thick bright wall.
3. Merge the two faces of each wall and bridge door-sized gaps.
4. Drop pieces that never touch the outer shell (furniture).
5. Label the regions those barriers leave behind, and measure them in pixels.
6. Detect windows, balcony rails, and door leaves on the same render.
7. Draw an orthogonal plan. ``render_schematic`` does that last step.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.openings import detect_openings, filter_doors
from src.schematic import render_schematic

# Tuned on the isometric marketing renders in input_examples/.
# Lengths are fractions of the shorter image side unless noted.
_MIN_SEGMENT_FRAC = 0.038
_AXIS_TOLERANCE_DEG = 12.0
_WALL_CLUSTER_FRAC = 0.042
_DOOR_GAP_FRAC = 0.115
_ENDPOINT_EXTEND_FRAC = 0.02
_BARRIER_THICKNESS_FRAC = 0.02
_MIN_ROOM_AREA_FRAC = 0.02
_MIN_SUPPORT_PX = 8
_POLYGON_EPSILON_PX = 4.0


def extract_layout(image_bgr: np.ndarray, source_name: str) -> tuple[np.ndarray, dict]:
    """Return an annotated BGR image and a JSON-serializable layout dict."""
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("expected a color image")

    height, width = image_bgr.shape[:2]
    scale = float(min(height, width))
    plan, lightness, chroma = _apartment_mask(image_bgr)
    horizontal, vertical = _wall_segments(image_bgr, plan, lightness, chroma, scale)
    # Bridge gaps up to a doorway so a door does not merge two rooms, and
    # remember the gaps that are wide enough to draw as doors.
    door_gap = scale * _DOOR_GAP_FRAC
    tolerance = scale * _WALL_CLUSTER_FRAC
    min_opening = scale * 0.035
    horizontal, horizontal_doors = _merge_collinear(horizontal, door_gap, tolerance, min_opening)
    vertical, vertical_doors = _merge_collinear(vertical, door_gap, tolerance, min_opening)
    horizontal, vertical, horizontal_doors, vertical_doors = _keep_structural(
        horizontal, vertical, horizontal_doors, vertical_doors, plan, scale
    )
    horizontal_doors = filter_doors(lightness, chroma, horizontal_doors, scale, True)
    vertical_doors = filter_doors(lightness, chroma, vertical_doors, scale, False)
    windows, balconies = detect_openings(image_bgr, plan, scale)
    barrier = _raster_barrier(horizontal, vertical, plan, scale)
    rooms = _rooms_from_barrier(plan, barrier, lightness, chroma)

    total_area = int(sum(room["area_px"] for room in rooms)) or 1
    for room in rooms:
        room["relative_area"] = round(room["area_px"] / total_area, 4)

    annotated, features = render_schematic(
        (height, width),
        plan,
        horizontal,
        vertical,
        horizontal_doors,
        vertical_doors,
        rooms,
        scale,
        windows,
        balconies,
    )
    for room in rooms:
        room.pop("_mask", None)
    result = {
        "source_image": source_name,
        "image_width": width,
        "image_height": height,
        "coordinate_system": "origin at the top-left corner, x to the right, y down",
        "area_unit": "pixels",
        "plan_area_px": int(plan.sum()),
        "total_room_area_px": int(sum(room["area_px"] for room in rooms)),
        "features": features,
        "rooms": rooms,
        "assumptions": [
            "The render is a near-orthographic top-down or axonometric view with axis-aligned walls.",
            "Walls are brighter and less saturated than most floors.",
            "Door openings are narrower than about 12 percent of the shorter image side.",
            "A window is two bright frame lines on an exterior wall with glass between them.",
            "A balcony is a dark rail, or a light rail with repeated balusters, on the exterior.",
            "Areas are pixel counts in image space. relative_area is each room's share of the detected room pixels.",
            "Furniture, fixtures, and other marks inside a room are not walls and are not drawn.",
        ],
    }
    return annotated, result


def _apartment_mask(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Largest non-white blob, with holes filled. Caption text stays outside it."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)
    background = (saturation < 12) & (value > 248)
    foreground = cv2.morphologyEx(
        (~background).astype(np.uint8),
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
    )
    plan = _largest_component(foreground.astype(bool))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed = cv2.morphologyEx(plan.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
    flood = closed.copy()
    flood_mask = np.zeros((closed.shape[0] + 2, closed.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 128)
    plan = (closed > 0) | (flood == 0)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0].astype(np.float32)
    chroma = np.sqrt(
        (lab[:, :, 1].astype(np.float32) - 128.0) ** 2
        + (lab[:, :, 2].astype(np.float32) - 128.0) ** 2
    )
    return plan, lightness, chroma


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return np.zeros(mask.shape, dtype=bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def _wall_segments(image_bgr, plan, lightness, chroma, scale: float):
    """Long horizontal and vertical edges that sit on a thick bright wall."""
    gray = cv2.GaussianBlur(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = detector.detect(gray)[0]
    min_length = scale * _MIN_SEGMENT_FRAC
    horizontal: list[tuple[float, float, float]] = []
    vertical: list[tuple[float, float, float]] = []
    if detected is None:
        return horizontal, vertical

    height, width = plan.shape
    for segment in detected:
        x1, y1, x2, y2 = (float(value) for value in np.asarray(segment).reshape(-1)[:4])
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < min_length:
            continue
        angle = abs(float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))) % 180.0
        is_horizontal = min(angle, 180.0 - angle) <= _AXIS_TOLERANCE_DEG
        is_vertical = abs(angle - 90.0) <= _AXIS_TOLERANCE_DEG
        if not (is_horizontal or is_vertical):
            continue

        samples = max(8, int(length / 4))
        ticks = np.linspace(0.05, 0.95, samples)
        xs = np.clip((x1 + (x2 - x1) * ticks).astype(int), 0, width - 1)
        ys = np.clip((y1 + (y2 - y1) * ticks).astype(int), 0, height - 1)
        if float(plan[ys, xs].mean()) < 0.5:
            continue
        if float(np.median(lightness[ys, xs])) < 148 or float(np.median(chroma[ys, xs])) > 20:
            continue
        if _support_width(xs, ys, is_horizontal, lightness, chroma) < _MIN_SUPPORT_PX:
            continue

        if is_horizontal:
            horizontal.append((float(np.median(ys)), min(x1, x2), max(x1, x2)))
        else:
            vertical.append((float(np.median(xs)), min(y1, y2), max(y1, y2)))
    return horizontal, vertical


def _support_width(xs, ys, horizontal: bool, lightness, chroma, max_offset: int = 28) -> float:
    """How many pixels of bright, low-chroma wall sit across the segment."""
    height, width = lightness.shape
    widths = []
    step = max(1, len(xs) // 12)
    for x, y in zip(xs[::step], ys[::step]):
        values = []
        for offset in range(-max_offset, max_offset + 1):
            yy, xx = (y + offset, x) if horizontal else (y, x + offset)
            on_wall = (
                0 <= yy < height
                and 0 <= xx < width
                and lightness[yy, xx] >= 155
                and chroma[yy, xx] <= 18
            )
            values.append(1 if on_wall else 0)
        center = max_offset
        if not values[center]:
            widths.append(0)
            continue
        low = center
        while low > 0 and values[low - 1]:
            low -= 1
        high = center
        while high < len(values) - 1 and values[high + 1]:
            high += 1
        widths.append(high - low + 1)
    if not widths:
        return 0.0
    return float(np.median(widths))


def _merge_collinear(segments, gap: float, cluster_tolerance: float, min_opening: float):
    """Collapse double wall faces and record door-sized gaps on the same line.

    Each segment is (transverse, start, end): y/x0/x1 for horizontal walls and
    x/y0/y1 for vertical walls. Openings are (transverse, start, end) gaps that
    are wide enough to draw as doors but still bridged for room separation.
    """
    if not segments:
        return [], []
    ordered = sorted(segments, key=lambda item: item[0])
    clusters = [[ordered[0]]]
    for transverse, start, end in ordered[1:]:
        cluster_center = float(np.median([item[0] for item in clusters[-1]]))
        if abs(transverse - cluster_center) > cluster_tolerance:
            clusters.append([(transverse, start, end)])
        else:
            clusters[-1].append((transverse, start, end))

    merged = []
    openings = []
    for cluster in clusters:
        transverse = float(np.median([item[0] for item in cluster]))
        intervals = sorted((min(start, end), max(start, end)) for _, start, end in cluster)
        current_start, current_end = intervals[0]
        for start, end in intervals[1:]:
            gap_size = start - current_end
            if gap_size <= gap:
                if gap_size >= min_opening:
                    openings.append((transverse, current_end, start))
                current_end = max(current_end, end)
            else:
                merged.append((transverse, current_start, current_end))
                current_start, current_end = start, end
        merged.append((transverse, current_start, current_end))
    return merged, openings


def _draw_segments(canvas, horizontal, vertical, scale: float) -> None:
    extension = scale * _ENDPOINT_EXTEND_FRAC
    thickness = max(11, int(round(scale * _BARRIER_THICKNESS_FRAC)))
    height, width = canvas.shape[:2]

    def clip(value, limit):
        return int(max(0, min(limit - 1, round(value))))

    for y, x0, x1 in horizontal:
        cv2.line(
            canvas,
            (clip(x0 - extension, width), clip(y, height)),
            (clip(x1 + extension, width), clip(y, height)),
            255,
            thickness,
        )
    for x, y0, y1 in vertical:
        cv2.line(
            canvas,
            (clip(x, width), clip(y0 - extension, height)),
            (clip(x, width), clip(y1 + extension, height)),
            255,
            thickness,
        )


def _keep_structural(horizontal, vertical, horizontal_doors, vertical_doors, plan, scale: float):
    """Drop wall pieces that never meet the outer shell (usually furniture)."""
    height, width = plan.shape
    barrier = np.zeros((height, width), np.uint8)
    _draw_segments(barrier, horizontal, vertical, scale)
    contours, _ = cv2.findContours(plan.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outer = np.zeros((height, width), np.uint8)
    cv2.drawContours(outer, contours, -1, 255, 4)
    touch = cv2.dilate(outer, np.ones((9, 9), np.uint8))
    count, labels, _, _ = cv2.connectedComponentsWithStats((barrier > 0).astype(np.uint8), 8)
    structural = np.zeros_like(barrier)
    for label in range(1, count):
        component = labels == label
        if np.any(component & (touch > 0)):
            structural[component] = 255

    def on_structure(transverse, start, end, horizontal_segment: bool) -> bool:
        midpoint_a = (start + end) / 2.0
        if horizontal_segment:
            x = int(np.clip(round(midpoint_a), 0, width - 1))
            y = int(np.clip(round(transverse), 0, height - 1))
        else:
            x = int(np.clip(round(transverse), 0, width - 1))
            y = int(np.clip(round(midpoint_a), 0, height - 1))
        return bool(structural[y, x])

    horizontal = [segment for segment in horizontal if on_structure(*segment, True)]
    vertical = [segment for segment in vertical if on_structure(*segment, False)]
    horizontal_doors = [door for door in horizontal_doors if on_structure(*door, True)]
    vertical_doors = [door for door in vertical_doors if on_structure(*door, False)]
    return horizontal, vertical, horizontal_doors, vertical_doors


def _raster_barrier(horizontal, vertical, plan, scale: float) -> np.ndarray:
    height, width = plan.shape
    barrier = np.zeros((height, width), np.uint8)
    _draw_segments(barrier, horizontal, vertical, scale)
    contours, _ = cv2.findContours(plan.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outer = np.zeros((height, width), np.uint8)
    cv2.drawContours(outer, contours, -1, 255, 4)
    return cv2.bitwise_or(barrier, outer)


def _rooms_from_barrier(plan, barrier, lightness, chroma) -> list[dict]:
    interior = (plan & (barrier == 0)).astype(np.uint8)
    interior = cv2.morphologyEx(interior, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(interior, 8)
    min_area = float(plan.sum()) * _MIN_ROOM_AREA_FRAC
    raw = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        box_w = int(stats[label, cv2.CC_STAT_WIDTH])
        box_h = int(stats[label, cv2.CC_STAT_HEIGHT])
        aspect = max(box_w, box_h) / max(1.0, min(box_w, box_h))
        component = _fill_holes(labels == label)
        area = int(component.sum())
        median_l = float(np.median(lightness[component]))
        median_c = float(np.median(chroma[component]))
        # Isometric wall tops are thin, very bright, and weakly colored.
        if aspect >= 3.2 and median_l > 190 and median_c < 12 and area < plan.sum() * 0.08:
            continue
        raw.append(component)

    raw.sort(key=lambda mask: int(mask.sum()), reverse=True)
    rooms = []
    for index, component in enumerate(raw, start=1):
        polygon = _polygon(component)
        if len(polygon) < 3:
            continue
        moments = cv2.moments(component.astype(np.uint8))
        if moments["m00"] == 0:
            continue
        centroid = [
            round(moments["m10"] / moments["m00"], 1),
            round(moments["m01"] / moments["m00"], 1),
        ]
        ys, xs = np.where(component)
        rooms.append(
            {
                "id": f"room_{index}",
                "polygon": polygon,
                "area_px": int(component.sum()),
                "centroid": centroid,
                "bbox": [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
                "_mask": component,
            }
        )
    return rooms


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    uint8 = mask.astype(np.uint8) * 255
    flood = uint8.copy()
    flood_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = flood == 0
    return mask | holes


def _polygon(mask: np.ndarray) -> list[list[int]]:
    smoothed = cv2.morphologyEx(
        mask.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    contours, _ = cv2.findContours(smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    approximate = cv2.approxPolyDP(contour, _POLYGON_EPSILON_PX, True)
    return [[int(point[0][0]), int(point[0][1])] for point in approximate]



