"""Find windows, balcony rails, and door leaves on a floor-plan render.

A segment is ``(orientation, transverse, start, end)``:

- ``"h"`` is a horizontal line. ``transverse`` is y, and ``start``/``end`` are x.
- ``"v"`` is a vertical line. ``transverse`` is x, and ``start``/``end`` are y.

Line segments are detected once per image and reused. A mark is kept only when
it has the shape of that object. The plan draws it only then.
"""

from __future__ import annotations

import cv2
import numpy as np

# (orientation, transverse, start, end)
Segment = tuple[str, float, float, float]


def detect_openings(image_bgr: np.ndarray, plan: np.ndarray, scale: float) -> tuple[list[Segment], list[Segment]]:
    """Return ``(windows, balconies)`` in image coordinates."""
    scan = _Scan(image_bgr, plan, scale)
    rails = _dark_railings(scan)
    windows = _find_windows(scan, rails)
    balconies = rails + _light_railings(scan)
    # A glazed opening wins when a rail and a window describe the same span.
    balconies = [segment for segment in balconies if not _overlaps_any(segment, windows, scale * 0.04)]
    return windows, _merge_segments(balconies, scale * 0.05)


def filter_doors(
    lightness: np.ndarray,
    chroma: np.ndarray,
    doors: list[tuple[float, float, float]],
    scale: float,
    horizontal: bool,
) -> list[tuple[float, float, float]]:
    """Keep wall gaps that are door-width and are not still a solid wall."""
    kept = []
    for transverse, start, end in doors:
        width = end - start
        if not (scale * 0.03 <= width <= scale * 0.12):
            continue
        if _solid_wall(lightness, chroma, transverse, start, end, horizontal):
            continue
        kept.append((float(transverse), float(start), float(end)))
    return kept


class _Scan:
    """Shared line segments and color planes for one render."""

    def __init__(self, image_bgr: np.ndarray, plan: np.ndarray, scale: float) -> None:
        self.plan = plan
        self.scale = scale
        # LSD is stabler on a light blur. Baluster counts and Canny use the raw gray.
        self.gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(self.gray, (3, 3), 0)
        detected = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(blurred)[0]
        self.lines = [] if detected is None else detected
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        self.lightness = lab[:, :, 0]
        self.chroma = np.sqrt((lab[:, :, 1] - 128.0) ** 2 + (lab[:, :, 2] - 128.0) ** 2)
        self.edges = cv2.Canny(self.gray, 40, 120)
        boundary = cv2.morphologyEx(plan.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        reach = int(scale * 0.045) | 1
        self.near_boundary = cv2.dilate(boundary.astype(np.uint8), np.ones((reach, reach), np.uint8)) > 0
        self.inside = cv2.dilate(plan.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0


def _find_windows(scan: _Scan, rails: list[Segment]) -> list[Segment]:
    """Two bright exterior frame lines with a varied, bright strip of glass between them."""
    # The outer frame can be long (the wall head). The inner frame is the sill.
    outer = _collect_segments(scan, 0.03, 0.42, boundary_only=True)
    inner = _collect_segments(scan, 0.03, 0.32, boundary_only=False)
    scale = scan.scale
    candidates = []
    for first in outer:
        for second in inner:
            if first[0] != second[0]:
                continue
            gap = abs(first[1] - second[1])
            if not (scale * 0.02 <= gap <= scale * 0.095):
                continue
            left = max(first[2], second[2])
            right = min(first[3], second[3])
            overlap = right - left
            shorter = min(first[3] - first[2], second[3] - second[2])
            # Long enough to be glazing, not a picture frame or a tub rim.
            if overlap < max(scale * 0.10, 0.5 * shorter):
                continue
            transverse = (first[1] + second[1]) / 2.0
            proposal = (first[0], transverse, left, right)
            if _line_is_dark(scan, first) or _line_is_dark(scan, second):
                continue
            if _overlaps_any(proposal, rails, scale * 0.05):
                continue
            if not _bright_glass(scan, proposal, gap):
                continue
            if not _on_exterior(scan.plan, first, second, left, right):
                continue
            # The strip itself has to open onto the outside on one side.
            if (
                _outside_fraction(scan.plan, first[0], transverse, left, right, gap) < 0.35
                and _outside_fraction(scan.plan, first[0], transverse, left, right, -gap) < 0.35
            ):
                continue
            candidates.append((*proposal, overlap))

    candidates.sort(key=lambda item: item[4], reverse=True)
    kept: list[Segment] = []
    for orient, transverse, start, end, _overlap in candidates:
        proposal = (orient, transverse, start, end)
        if _overlaps_any(proposal, kept, scale * 0.03):
            continue
        kept.append(proposal)
    return kept


def _dark_railings(scan: _Scan) -> list[Segment]:
    """Thin dark components on the exterior. These are metal balcony rails."""
    kernel = int(scan.scale * 0.03) | 1
    near = cv2.dilate(
        (cv2.morphologyEx(scan.plan.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0).astype(np.uint8),
        np.ones((kernel, kernel), np.uint8),
    ) > 0
    dark = ((scan.lightness < 80) & near & scan.plan).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    railings = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        box_w = int(stats[label, cv2.CC_STAT_WIDTH])
        box_h = int(stats[label, cv2.CC_STAT_HEIGHT])
        long_side = max(box_w, box_h)
        short_side = max(1, min(box_w, box_h))
        if area < 80 or long_side < scan.scale * 0.08 or long_side / short_side < 3 or short_side > scan.scale * 0.06:
            continue
        ys, xs = np.nonzero(labels == label)
        if box_w >= box_h:
            railings.append(("h", float(np.median(ys)), float(xs.min()), float(xs.max())))
        else:
            railings.append(("v", float(np.median(xs)), float(ys.min()), float(ys.max())))
    return railings


def _light_railings(scan: _Scan) -> list[Segment]:
    """A pale rail is an exterior line with a repeating row of balusters just inside it."""
    railings = []
    for orient, transverse, start, end in _collect_segments(scan, 0.08, 0.28, boundary_only=True):
        outward = _outward_offset(scan.plan, orient, transverse, start, end)
        if _outside_fraction(scan.plan, orient, transverse, start, end, outward) < 0.45:
            continue
        if _baluster_runs(scan.gray, orient, transverse, start, end, -outward) >= 5:
            railings.append((orient, transverse, start, end))
    return railings


def _collect_segments(scan: _Scan, min_frac: float, max_frac: float, boundary_only: bool) -> list[Segment]:
    """Axis-aligned LSD segments, either on the footprint edge or anywhere inside it."""
    height, width = scan.plan.shape
    segments: list[Segment] = []
    for raw in scan.lines:
        x1, y1, x2, y2 = (float(value) for value in np.asarray(raw).reshape(-1)[:4])
        length = float(np.hypot(x2 - x1, y2 - y1))
        if not (scan.scale * min_frac <= length <= scan.scale * max_frac):
            continue
        angle = abs(float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))) % 180.0
        horizontal = min(angle, 180.0 - angle) <= 12.0
        vertical = abs(angle - 90.0) <= 12.0
        if not (horizontal or vertical):
            continue
        mid_x = int(round((x1 + x2) / 2.0))
        mid_y = int(round((y1 + y2) / 2.0))
        if not (0 <= mid_x < width and 0 <= mid_y < height):
            continue
        on_edge = bool(scan.near_boundary[mid_y, mid_x])
        if boundary_only and not on_edge:
            continue
        if not boundary_only and not scan.inside[mid_y, mid_x]:
            continue
        if horizontal:
            segments.append(("h", float(np.median((y1, y2))), min(x1, x2), max(x1, x2)))
        else:
            segments.append(("v", float(np.median((x1, x2))), min(y1, y2), max(y1, y2)))
    return segments


def _bright_glass(scan: _Scan, segment: Segment, gap: float) -> bool:
    """The strip between the frames is bright, pale, textured, and not a flat wall face."""
    orient, transverse, start, end = segment
    lightness, chroma = _sample_strip(scan, orient, transverse, start, end, gap * 0.35, 12, 5)
    if len(lightness) < 8:
        return False
    bright = float(np.median(lightness)) > 155 and float(np.median(chroma)) < 35
    not_a_rail = float(np.mean(np.array(lightness) < 80)) < 0.12
    # A tighter band than the brightness check. The wider band reaches furniture
    # beside a flat wall and would treat that wall as glass.
    varied, _chroma = _sample_strip(scan, orient, transverse, start, end, gap * 0.3, 16, 5)
    textured = float(np.std(varied)) >= 10 if len(varied) >= 8 else False
    textured = textured and _edge_fraction(scan, orient, transverse, start, end, gap) >= 0.09
    return bright and not_a_rail and textured


def _edge_fraction(scan: _Scan, orient: str, transverse: float, start: float, end: float, gap: float) -> float:
    hits = 0
    total = 0
    for along in np.linspace(start, end, 40):
        for offset in np.linspace(-gap * 0.4, gap * 0.4, 7):
            x, y = _at(orient, transverse, along, offset)
            if _in(scan.edges, x, y):
                total += 1
                hits += int(scan.edges[y, x] > 0)
    return hits / max(1, total)


def _sample_strip(scan: _Scan, orient, transverse, start, end, half_width, along_count, across_count):
    lightness, chroma = [], []
    for along in np.linspace(start, end, along_count):
        for offset in np.linspace(-half_width, half_width, across_count):
            x, y = _at(orient, transverse, along, offset)
            if _in(scan.lightness, x, y):
                lightness.append(float(scan.lightness[y, x]))
                chroma.append(float(scan.chroma[y, x]))
    return lightness, chroma


def _line_is_dark(scan: _Scan, segment: Segment) -> bool:
    orient, transverse, start, end = segment
    values = []
    for along in np.linspace(start, end, 12):
        x, y = _at(orient, transverse, along, 0)
        if _in(scan.lightness, x, y):
            values.append(float(scan.lightness[y, x]))
    return bool(values) and float(np.median(values)) < 110


def _on_exterior(plan, first: Segment, second: Segment, start: float, end: float) -> bool:
    """One of the two frames has to sit on the outside of the apartment."""
    for segment, other in ((first, second), (second, first)):
        direction = 12.0 if segment[1] >= other[1] else -12.0
        if _outside_fraction(plan, segment[0], segment[1], start, end, direction) >= 0.35:
            return True
    return False


def _baluster_runs(gray, orient: str, transverse: float, start: float, end: float, inward: float) -> int:
    """How many dark slats cross the rail. ``inward`` is ±8, the step off the line."""
    height, width = gray.shape
    best = 0
    direction = 1.0 if inward >= 0 else -1.0
    for distance in (8, 14, 22):
        samples = []
        for along in np.linspace(start, end, max(8, int(end - start))):
            x, y = _at(orient, transverse, along, direction * distance)
            if 0 <= x < width and 0 <= y < height:
                samples.append(float(gray[y, x]))
        if len(samples) < 8:
            continue
        row = np.array(samples, np.float32)
        dark = row < (float(np.median(row)) - 8.0)
        runs = int(np.sum(dark & np.concatenate(([True], ~dark[:-1]))))
        best = max(best, runs)
    return best


def _outward_offset(plan, orient: str, transverse: float, start: float, end: float) -> float:
    """±8 pixels, toward the side of the line that leaves the apartment."""
    positive = _outside_fraction(plan, orient, transverse, start, end, 8)
    negative = _outside_fraction(plan, orient, transverse, start, end, -8)
    return 8.0 if positive >= negative else -8.0


def _solid_wall(lightness, chroma, transverse, start, end, horizontal: bool) -> bool:
    """True when a supposed door gap is still uniform bright wall."""
    height, width = lightness.shape
    samples_l = []
    samples_c = []
    for position in np.linspace(start, end, 9):
        if horizontal:
            x = int(np.clip(round(position), 0, width - 1))
            y = int(np.clip(round(transverse), 0, height - 1))
        else:
            x = int(np.clip(round(transverse), 0, width - 1))
            y = int(np.clip(round(position), 0, height - 1))
        samples_l.append(lightness[y, x])
        samples_c.append(chroma[y, x])
    if not samples_l:
        return False
    return float(np.median(samples_l)) > 185 and float(np.median(samples_c)) < 18 and float(np.std(samples_l)) < 12


def _outside_fraction(plan, orient: str, transverse: float, start: float, end: float, offset: float) -> float:
    height, width = plan.shape
    outside = 0
    total = 0
    for along in np.linspace(start, end, 15):
        x, y = _at(orient, transverse, along, offset)
        total += 1
        if not (0 <= x < width and 0 <= y < height and plan[y, x]):
            outside += 1
    return outside / max(1, total)


def _overlaps_any(segment: Segment, others: list[Segment], tolerance: float) -> bool:
    orient, transverse, start, end = segment
    for other, other_t, other_a, other_b in others:
        if other != orient or abs(other_t - transverse) > tolerance:
            continue
        overlap = min(end, other_b) - max(start, other_a)
        if overlap > 0.4 * min(end - start, other_b - other_a):
            return True
    return False


def _merge_segments(segments: list[Segment], tolerance: float) -> list[Segment]:
    merged: list[Segment] = []
    for orient, transverse, start, end in segments:
        placed = False
        for index, (other, other_t, other_a, other_b) in enumerate(merged):
            if other != orient or abs(other_t - transverse) > tolerance:
                continue
            if start <= other_b + tolerance and end >= other_a - tolerance:
                merged[index] = (orient, (other_t + transverse) / 2.0, min(start, other_a), max(end, other_b))
                placed = True
                break
        if not placed:
            merged.append((orient, transverse, start, end))
    return merged


def _at(orient: str, transverse: float, along: float, offset: float) -> tuple[int, int]:
    """Pixel at ``along`` the segment and ``offset`` pixels to one side."""
    if orient == "h":
        return int(round(along)), int(round(transverse + offset))
    return int(round(transverse + offset)), int(round(along))


def _in(array: np.ndarray, x: int, y: int) -> bool:
    height, width = array.shape[:2]
    return 0 <= x < width and 0 <= y < height
