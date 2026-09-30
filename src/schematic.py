"""Draw detected wall lines as a black-and-white plan.

Each room is labeled with its pixel area. Windows, balcony rails, and doors
are drawn only when those objects were detected. Furniture and room names are not.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
)


def render_schematic(
    shape,
    plan,
    horizontal,
    vertical,
    horizontal_doors,
    vertical_doors,
    rooms,
    scale: float,
    windows=(),
    balconies=(),
) -> tuple[np.ndarray, list[dict]]:
    """Draw the plan in the source image's pixel frame.

    Returns the BGR canvas and the window, door, and balcony spans that were
    actually drawn. Coordinates stay in the source image so they match the JSON.
    """
    height, width = shape
    exterior = max(8, int(round(scale * 0.022)))
    interior = max(5, int(round(scale * 0.012)))
    # Square the footprint first. Interior walls are then snapped onto it so
    # every joint is a corner instead of a slanted isometric edge.
    shell = _blocky_plan(plan, scale)
    if int(shell.sum()) < 100:
        shell = plan
    outline = _orthogonal_outline(shell, scale)
    if outline is not None:
        raster = np.zeros(shell.shape, np.uint8)
        cv2.fillPoly(raster, [outline], 1)
        shell = raster > 0
    labels = np.zeros(shell.shape, np.int32)
    for index, room in enumerate(rooms, start=1):
        labels[room["_mask"]] = index
    offset = max(interior + 4, int(round(scale * 0.03)))
    axis_tol = max(12.0, scale * 0.036)
    join = max(20.0, scale * 0.09)
    horizontal, vertical, horizontal_doors, vertical_doors = _align_walls(
        horizontal,
        vertical,
        horizontal_doors,
        vertical_doors,
        outline,
        labels,
        offset,
        axis_tol,
        join,
    )

    canvas = np.full((height, width, 3), 255, np.uint8)
    kernel = exterior if exterior % 2 == 1 else exterior + 1
    floor = cv2.erode(shell.astype(np.uint8), np.ones((kernel, kernel), np.uint8)) > 0
    canvas[shell] = (0, 0, 0)
    canvas[floor] = (255, 255, 255)

    _paint_interior_walls(canvas, horizontal, vertical, shell, interior, outline)
    # Door gaps are cut out of the wall. The swing is drawn afterwards, into the floor.
    _punch_openings(canvas, horizontal_doors, vertical_doors, horizontal, vertical, interior, shell)
    features = []
    features.extend(_draw_openings(canvas, windows, "window", outline, shell, exterior, scale))
    features.extend(_draw_openings(canvas, balconies, "balcony", outline, shell, exterior, scale))
    features.extend(_draw_doors(canvas, horizontal_doors, vertical_doors, interior, scale, floor))
    canvas[~shell] = (255, 255, 255)
    _draw_areas(canvas, rooms, scale)
    return canvas, features


def _separates_rooms(segment, horizontal: bool, labels: np.ndarray, offset: int) -> bool:
    """True when the two sides of a segment fall in different rooms."""
    transverse, start, end = segment
    height, width = labels.shape
    separated = 0
    checked = 0
    for position in np.linspace(start, end, 11):
        if horizontal:
            x = int(round(position))
            y = int(round(transverse))
            y_a, y_b = y - offset, y + offset
            if not (0 <= x < width and 0 <= y_a < height and 0 <= y_b < height):
                continue
            side_a, side_b = int(labels[y_a, x]), int(labels[y_b, x])
        else:
            y = int(round(position))
            x = int(round(transverse))
            x_a, x_b = x - offset, x + offset
            if not (0 <= y < height and 0 <= x_a < width and 0 <= x_b < width):
                continue
            side_a, side_b = int(labels[y, x_a]), int(labels[y, x_b])
        checked += 1
        if side_a > 0 and side_b > 0 and side_a != side_b:
            separated += 1
    return checked > 0 and separated / checked >= 0.35


def _blocky_plan(plan: np.ndarray, scale: float) -> np.ndarray:
    """Straighten the footprint so the outer wall reads as an orthogonal plan."""
    kernel_size = max(9, int(round(scale * 0.028)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    block = cv2.morphologyEx(plan.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
    block = cv2.morphologyEx(block, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((block > 0).astype(np.uint8), 8)
    if count <= 1:
        return plan
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def _paint_interior_walls(canvas, horizontal, vertical, shell, thickness: int, outline) -> None:
    """Draw square-ended bars that meet at the joint and stop on the outer wall."""
    height, width = shell.shape
    half = thickness // 2
    outer_h, outer_v = _outline_edges(outline)

    def on_outer(end: float, transverse: float, outers) -> bool:
        return any(abs(end - edge) <= 3 and low - 3 <= transverse <= high + 3 for edge, low, high in outers)

    def bar(x0, y0, x1, y1):
        x0, x1 = sorted((int(max(0, x0)), int(min(width - 1, x1))))
        y0, y1 = sorted((int(max(0, y0)), int(min(height - 1, y1))))
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 0, 0), -1)

    for transverse, start, end in horizontal:
        y = int(round(transverse))
        left, right = min(start, end), max(start, end)
        x0 = int(round(left)) if on_outer(left, transverse, outer_v) else int(round(left)) - half
        x1 = int(round(right)) if on_outer(right, transverse, outer_v) else int(round(right)) + half
        bar(x0, y - half, x1, y - half + thickness - 1)
    for transverse, start, end in vertical:
        x = int(round(transverse))
        top, bottom = min(start, end), max(start, end)
        y0 = int(round(top)) if on_outer(top, transverse, outer_h) else int(round(top)) - half
        y1 = int(round(bottom)) if on_outer(bottom, transverse, outer_h) else int(round(bottom)) + half
        bar(x - half, y0, x - half + thickness - 1, y1)

    canvas[~shell] = (255, 255, 255)


def _orthogonal_outline(mask: np.ndarray, scale: float):
    """Replace slanted outline edges with horizontal and vertical corners."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    epsilon = max(6.0, 0.008 * cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(approx) < 3:
        return None
    points = _manhattan_ring(approx)
    if len(points) < 4:
        return None
    raster = np.zeros(mask.shape, np.uint8)
    cv2.fillPoly(raster, [np.array(points, np.int32)], 1)
    overlap = float(np.logical_and(raster > 0, mask).sum())
    union = float(np.logical_or(raster > 0, mask).sum())
    recall = overlap / max(1.0, float(mask.sum()))
    if union == 0 or overlap / union < 0.7 or recall < 0.9:
        return None
    return np.array(points, np.int32)


def _manhattan_ring(approx: np.ndarray) -> list[tuple[int, int]]:
    """Walk the simplified contour, inserting a corner wherever an edge is diagonal."""
    ring: list[tuple[int, int]] = []
    count = len(approx)
    for index in range(count):
        x1, y1 = (int(approx[index][0]), int(approx[index][1]))
        x2, y2 = (int(approx[(index + 1) % count][0]), int(approx[(index + 1) % count][1]))
        if abs(x2 - x1) >= abs(y2 - y1):
            elbow = (x2, y1)
        else:
            elbow = (x1, y2)
        for point in ((x1, y1), elbow, (x2, y2)):
            if not ring or abs(point[0] - ring[-1][0]) + abs(point[1] - ring[-1][1]) > 2:
                ring.append(point)
    if len(ring) > 1 and abs(ring[0][0] - ring[-1][0]) + abs(ring[0][1] - ring[-1][1]) <= 2:
        ring.pop()
    return _simplify_ortho(ring)


def _simplify_ortho(points: list[tuple[int, int]], tolerance: int = 6) -> list[tuple[int, int]]:
    """Share one coordinate across nearly aligned corners, then drop colinear points."""

    def canon(values: list[int]) -> dict[int, int]:
        ordered = sorted(set(values))
        groups: list[list[int]] = []
        for value in ordered:
            if not groups or value - groups[-1][-1] > tolerance:
                groups.append([value])
            else:
                groups[-1].append(value)
        mapping = {}
        for group in groups:
            center = int(round(sum(group) / len(group)))
            for value in group:
                mapping[value] = center
        return mapping

    if len(points) < 4:
        return points
    mapped_x = canon([point[0] for point in points])
    mapped_y = canon([point[1] for point in points])
    snapped = [(mapped_x[point[0]], mapped_y[point[1]]) for point in points]
    cleaned = [snapped[0]]
    for point in snapped[1:]:
        if point == cleaned[-1]:
            continue
        if len(cleaned) >= 2:
            older = cleaned[-2]
            if older[0] == cleaned[-1][0] == point[0] or older[1] == cleaned[-1][1] == point[1]:
                cleaned[-1] = point
                continue
        cleaned.append(point)
    if len(cleaned) > 2 and (cleaned[0][0] == cleaned[-1][0] == cleaned[-2][0] or cleaned[0][1] == cleaned[-1][1] == cleaned[-2][1]):
        cleaned.pop()
    return cleaned


def _align_walls(horizontal, vertical, horizontal_doors, vertical_doors, outline, labels, offset, axis_tol, join):
    """Snap walls onto shared axes and stop each one on the wall it meets."""
    outer_h, outer_v = _outline_edges(outline)
    horizontal = [wall for wall in horizontal if _separates_rooms(wall, True, labels, offset)]
    vertical = [wall for wall in vertical if _separates_rooms(wall, False, labels, offset)]
    horizontal = _snap_axis(horizontal, [edge[0] for edge in outer_h], axis_tol)
    vertical = _snap_axis(vertical, [edge[0] for edge in outer_v], axis_tol)
    horizontal = _extend_to_stops(horizontal, vertical + outer_v, join)
    vertical = _extend_to_stops(vertical, horizontal + outer_h, join)
    horizontal, vertical = _split_crossings(horizontal, vertical)
    horizontal = [wall for wall in horizontal if _separates_rooms(wall, True, labels, offset) and wall[2] - wall[1] > offset]
    vertical = [wall for wall in vertical if _separates_rooms(wall, False, labels, offset) and wall[2] - wall[1] > offset]
    horizontal = _extend_to_stops(horizontal, vertical + outer_v, join)
    vertical = _extend_to_stops(vertical, horizontal + outer_h, join)
    horizontal_doors = _snap_doors(horizontal_doors, horizontal, axis_tol)
    vertical_doors = _snap_doors(vertical_doors, vertical, axis_tol)
    return horizontal, vertical, horizontal_doors, vertical_doors


def _outline_edges(outline):
    if outline is None or len(outline) < 3:
        return [], []
    points = [(int(point[0]), int(point[1])) for point in outline.reshape(-1, 2)]
    horizontal, vertical = [], []
    for start, end in zip(points, points[1:] + points[:1]):
        x1, y1 = start
        x2, y2 = end
        if abs(y1 - y2) <= 2 and abs(x2 - x1) > 2:
            horizontal.append((float(y1), float(min(x1, x2)), float(max(x1, x2))))
        elif abs(x1 - x2) <= 2 and abs(y2 - y1) > 2:
            vertical.append((float(x1), float(min(y1, y2)), float(max(y1, y2))))
    return horizontal, vertical


def _snap_axis(segments, guides, tolerance):
    """Pull nearby parallel walls onto one shared coordinate."""
    if not segments:
        return []
    ordered = sorted(segments, key=lambda item: item[0])
    groups = [[ordered[0]]]
    for segment in ordered[1:]:
        center = float(np.median([item[0] for item in groups[-1]]))
        if abs(segment[0] - center) <= tolerance:
            groups[-1].append(segment)
        else:
            groups.append([segment])
    snapped = []
    for group in groups:
        center = float(np.median([item[0] for item in group]))
        guide = [value for value in guides if abs(value - center) <= tolerance]
        if guide:
            center = float(min(guide, key=lambda value: abs(value - center)))
        for _, start, end in group:
            snapped.append((center, float(min(start, end)), float(max(start, end))))
    return snapped


def _extend_to_stops(segments, stops, tolerance):
    """Move each endpoint onto the nearest crossing wall, and no farther."""
    aligned = []
    for transverse, start, end in segments:
        covering = []
        for stop, low, high in stops:
            if low - tolerance <= transverse <= high + tolerance:
                covering.append(stop)
        start = _snap_endpoint(start, covering, tolerance)
        end = _snap_endpoint(end, covering, tolerance)
        if end - start > 4:
            aligned.append((transverse, start, end))
    return aligned


def _snap_endpoint(position, stops, tolerance):
    if not stops:
        return position
    nearby = [stop for stop in stops if abs(stop - position) <= tolerance]
    if not nearby:
        return position
    return float(min(nearby, key=lambda stop: abs(stop - position)))


def _split_crossings(horizontal, vertical):
    """Break a wall where another wall crosses it so the pieces meet at that corner."""
    split_h = []
    for transverse, start, end in horizontal:
        cuts = [start, end]
        for other, low, high in vertical:
            if start + 2 < other < end - 2 and low - 2 <= transverse <= high + 2:
                cuts.append(other)
        split_h.extend(_pieces(transverse, cuts))
    split_v = []
    for transverse, start, end in vertical:
        cuts = [start, end]
        for other, low, high in horizontal:
            if start + 2 < other < end - 2 and low - 2 <= transverse <= high + 2:
                cuts.append(other)
        split_v.extend(_pieces(transverse, cuts))
    return split_h, split_v


def _pieces(transverse, cuts):
    ordered = sorted(set(round(cut, 2) for cut in cuts))
    return [(transverse, ordered[index], ordered[index + 1]) for index in range(len(ordered) - 1) if ordered[index + 1] - ordered[index] > 4]


def _snap_doors(doors, walls, tolerance):
    snapped = []
    for transverse, start, end in doors:
        matches = [
            wall
            for wall in walls
            if abs(wall[0] - transverse) <= tolerance and wall[1] - 4 <= start and end <= wall[2] + 4
        ]
        if not matches:
            continue
        wall = min(matches, key=lambda item: abs(item[0] - transverse))
        snapped.append((wall[0], max(start, wall[1]), min(end, wall[2])))
    return snapped


def _punch_openings(canvas, horizontal_doors, vertical_doors, horizontal, vertical, thickness: int, shell) -> None:
    """Open a door without erasing the corner where its wall meets another wall."""
    gap_pad = max(2, thickness // 2)
    margin = thickness
    height, width = shell.shape

    def wipe(x0, y0, x1, y1):
        x0, x1 = sorted((int(max(0, x0)), int(min(width - 1, x1))))
        y0, y1 = sorted((int(max(0, y0)), int(min(height - 1, y1))))
        canvas[y0 : y1 + 1, x0 : x1 + 1] = (255, 255, 255)

    def inset(start, end, host):
        if host is None:
            return start, end
        return max(start, host[1] + margin), min(end, host[2] - margin)

    for y, x0, x1 in horizontal_doors:
        host = next((wall for wall in horizontal if abs(wall[0] - y) <= 2 and wall[1] - 2 <= x0 and x1 <= wall[2] + 2), None)
        a, b = inset(x0, x1, host)
        if b - a >= 6:
            wipe(a, y - gap_pad, b, y + gap_pad)
    for x, y0, y1 in vertical_doors:
        host = next((wall for wall in vertical if abs(wall[0] - x) <= 2 and wall[1] - 2 <= y0 and y1 <= wall[2] + 2), None)
        a, b = inset(y0, y1, host)
        if b - a >= 6:
            wipe(x - gap_pad, a, x + gap_pad, b)
    canvas[~shell] = (255, 255, 255)



def _draw_openings(canvas, segments, kind: str, outline, shell, exterior: int, scale: float) -> list[dict]:
    """Snap a detected window or railing onto the exterior wall and draw it there."""
    outer_h, outer_v = _outline_edges(outline)
    tolerance = max(18.0, scale * 0.07)
    features = []
    for orient, transverse, start, end in segments:
        edges = outer_h if orient == "h" else outer_v
        snapped = _snap_to_edges(transverse, start, end, edges, tolerance)
        if snapped is None:
            continue
        edge, span_start, span_end = snapped
        if kind == "window":
            _paint_window(canvas, shell, orient, edge, span_start, span_end, exterior)
        else:
            _paint_railing(canvas, shell, orient, edge, span_start, span_end, exterior)
        features.append(_feature(kind, orient, edge, span_start, span_end))
    return features


def _snap_to_edges(transverse, start, end, edges, tolerance):
    best = None
    for edge, edge_start, edge_end in edges:
        if abs(edge - transverse) > tolerance:
            continue
        left = max(start, edge_start)
        right = min(end, edge_end)
        overlap = right - left
        if overlap < max(12.0, 0.4 * (end - start)):
            continue
        distance = abs(edge - transverse)
        if best is None or distance < best[0]:
            best = (distance, edge, left, right)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _inward_sign(shell, orient: str, transverse: float, start: float, end: float) -> int:
    height, width = shell.shape
    mid = int(round((start + end) / 2.0))
    if orient == "h":
        y = int(np.clip(round(transverse), 0, height - 1))
        x = int(np.clip(mid, 0, width - 1))
        if shell[min(height - 1, y + 4), x] and not shell[max(0, y - 4), x]:
            return 1
        if shell[max(0, y - 4), x] and not shell[min(height - 1, y + 4), x]:
            return -1
    else:
        x = int(np.clip(round(transverse), 0, width - 1))
        y = int(np.clip(mid, 0, height - 1))
        if shell[y, min(width - 1, x + 4)] and not shell[y, max(0, x - 4)]:
            return 1
        if shell[y, max(0, x - 4)] and not shell[y, min(width - 1, x + 4)]:
            return -1
    return 1


def _paint_window(canvas, shell, orient, transverse, start, end, exterior: int) -> None:
    """Three thin glazing lines in a break in the exterior wall."""
    _clear_band(canvas, shell, orient, transverse, start, end, exterior)
    stroke = max(1, exterior // 12)
    _stroke_band(canvas, shell, orient, transverse, start, end, exterior, (0.25, 0.5, 0.75), stroke)


def _paint_railing(canvas, shell, orient, transverse, start, end, exterior: int) -> None:
    """Two thin lines in place of a solid wall, along a detected balcony rail."""
    _clear_band(canvas, shell, orient, transverse, start, end, exterior)
    stroke = max(1, exterior // 14)
    _stroke_band(canvas, shell, orient, transverse, start, end, exterior, (0.2, 0.55), stroke)


def _clear_band(canvas, shell, orient, transverse, start, end, depth: int) -> None:
    y0, y1, x0, x1 = _band_box(shell, orient, transverse, start, end, depth)
    band = shell[y0:y1, x0:x1]
    canvas[y0:y1, x0:x1][band] = (255, 255, 255)


def _stroke_band(canvas, shell, orient, transverse, start, end, depth: int, fractions, stroke: int) -> None:
    y0, y1, x0, x1 = _band_box(shell, orient, transverse, start, end, depth)
    if y1 <= y0 or x1 <= x0:
        return
    for fraction in fractions:
        if orient == "h":
            yy = int(round(y0 + (y1 - y0 - 1) * fraction))
            row = shell[yy, x0:x1]
            xs = np.flatnonzero(row)
            if len(xs):
                cv2.line(canvas, (x0 + int(xs[0]), yy), (x0 + int(xs[-1]), yy), (0, 0, 0), stroke, cv2.LINE_8)
        else:
            xx = int(round(x0 + (x1 - x0 - 1) * fraction))
            col = shell[y0:y1, xx]
            ys = np.flatnonzero(col)
            if len(ys):
                cv2.line(canvas, (xx, y0 + int(ys[0])), (xx, y0 + int(ys[-1])), (0, 0, 0), stroke, cv2.LINE_8)


def _band_box(shell, orient, transverse, start, end, depth: int):
    height, width = shell.shape
    sign = _inward_sign(shell, orient, transverse, start, end)
    if orient == "h":
        y_outer = int(np.clip(round(transverse), 0, height - 1))
        y_inner = int(np.clip(y_outer + sign * depth, 0, height - 1))
        y0, y1 = sorted((y_outer, y_inner))
        x0 = int(np.clip(round(min(start, end)), 0, width - 1))
        x1 = int(np.clip(round(max(start, end)), 0, width))
        return y0, min(height, y1 + 1), x0, min(width, x1 + 1)
    x_outer = int(np.clip(round(transverse), 0, width - 1))
    x_inner = int(np.clip(x_outer + sign * depth, 0, width - 1))
    x0, x1 = sorted((x_outer, x_inner))
    y0 = int(np.clip(round(min(start, end)), 0, height - 1))
    y1 = int(np.clip(round(max(start, end)), 0, height))
    return y0, min(height, y1 + 1), x0, min(width, x1 + 1)


def _draw_doors(canvas, horizontal_doors, vertical_doors, thickness: int, scale: float, floor) -> list[dict]:
    """Quarter-circle swing for each detected door, opening into the room."""
    stroke = max(1, thickness // 5)
    features = []
    for transverse, start, end in horizontal_doors:
        if _swing(canvas, "h", transverse, start, end, stroke, floor):
            features.append(_feature("door", "h", transverse, start, end))
    for transverse, start, end in vertical_doors:
        if _swing(canvas, "v", transverse, start, end, stroke, floor):
            features.append(_feature("door", "v", transverse, start, end))
    return features


def _swing(canvas, orient: str, transverse, start, end, stroke: int, floor) -> bool:
    height, width = canvas.shape[:2]
    gap = float(end - start)
    if gap < 8:
        return False
    radius = int(round(gap))
    if orient == "h":
        y = int(round(transverse))
        down = _open_pixels(int((start + end) / 2), y + radius, radius, floor)
        up = _open_pixels(int((start + end) / 2), y - radius, radius, floor)
        if down == 0 and up == 0:
            return False
        center = (int(round(start)), y)
        if down >= up:
            angles = (0, 90)
            leaf = (center[0], min(height - 1, center[1] + radius))
        else:
            angles = (270, 360)
            leaf = (center[0], max(0, center[1] - radius))
    else:
        x = int(round(transverse))
        right = _open_pixels(x + radius, int((start + end) / 2), radius, floor)
        left = _open_pixels(x - radius, int((start + end) / 2), radius, floor)
        if right == 0 and left == 0:
            return False
        center = (x, int(round(start)))
        if right >= left:
            angles = (0, 90)
            leaf = (min(width - 1, center[0] + radius), center[1])
        else:
            angles = (90, 180)
            leaf = (max(0, center[0] - radius), center[1])
    cv2.ellipse(canvas, center, (radius, radius), 0, angles[0], angles[1], (0, 0, 0), stroke, cv2.LINE_8)
    cv2.line(canvas, center, leaf, (0, 0, 0), stroke, cv2.LINE_8)
    return True


def _open_pixels(x: int, y: int, radius: int, floor) -> int:
    height, width = floor.shape
    x0, x1 = max(0, x - radius // 3), min(width, x + radius // 3)
    y0, y1 = max(0, y - radius // 3), min(height, y + radius // 3)
    if x1 <= x0 or y1 <= y0:
        return 0
    return int(floor[y0:y1, x0:x1].sum())


def _feature(kind: str, orient: str, transverse, start, end) -> dict:
    if orient == "h":
        return {
            "kind": kind,
            "orientation": "horizontal",
            "x0": int(round(min(start, end))),
            "y0": int(round(transverse)),
            "x1": int(round(max(start, end))),
            "y1": int(round(transverse)),
        }
    return {
        "kind": kind,
        "orientation": "vertical",
        "x0": int(round(transverse)),
        "y0": int(round(min(start, end))),
        "x1": int(round(transverse)),
        "y1": int(round(max(start, end))),
    }


def _draw_areas(canvas, rooms, scale: float) -> None:
    """Write each room's pixel area inside the room, and nothing else."""
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    for room in rooms:
        mask = room.get("_mask")
        if mask is None or int(mask.sum()) == 0:
            continue
        area_px = int(room["area_px"])
        share = float(room.get("relative_area") or 0.0)
        lines = [f"{area_px} px", f"{share:.0%}"]
        _paint_area(rgb, room, lines, scale)
    canvas[:] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _paint_area(rgb, room: dict, lines: list[str], scale: float) -> None:
    x, y, box_w, box_h = room["bbox"]
    mask = room["_mask"]
    size = max(11, int(round(scale * 0.022)))
    chosen = None
    while size >= 8:
        font = _font(size)
        probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        boxes = [probe.textbbox((0, 0), line, font=font) for line in lines]
        width = max(box[2] - box[0] for box in boxes)
        height = sum(box[3] - box[1] for box in boxes) + 2
        if width <= box_w * 0.9 and height <= box_h * 0.55:
            chosen = (font, boxes, width, height)
            break
        if len(lines) > 1 and size == 8:
            lines = lines[:1]
            size = max(11, int(round(scale * 0.022)))
            continue
        size -= 1
    if chosen is None:
        return
    font, boxes, width, height = chosen
    cx, cy = _label_anchor(room)
    origin_x = int(np.clip(cx - width / 2, x + 2, max(x + 2, x + box_w - width - 2)))
    origin_y = int(np.clip(cy - height / 2, y + 2, max(y + 2, y + box_h - height - 2)))
    overlay = Image.fromarray(np.full_like(rgb, 255))
    draw = ImageDraw.Draw(overlay)
    cursor = origin_y
    for line, box in zip(lines, boxes):
        line_w = box[2] - box[0]
        draw.text((origin_x + (width - line_w) / 2, cursor), line, fill=(0, 0, 0), font=font)
        cursor += box[3] - box[1] + 2
    ink = np.any(np.asarray(overlay) < 250, axis=2)
    if int(ink.sum()) == 0 or int((ink & mask).sum()) < 0.85 * int(ink.sum()):
        return
    rgb[ink & mask] = (0, 0, 0)


def _label_anchor(room: dict) -> tuple[float, float]:
    """Point deepest inside the room, so the caption does not sit on a wall."""
    mask = room.get("_mask")
    if mask is None or int(mask.sum()) == 0:
        return room["centroid"]
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    index = int(np.argmax(distance))
    yy, xx = np.unravel_index(index, distance.shape)
    return float(xx), float(yy)


def _font(size: int):
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).is_file():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
    return ImageFont.load_default()

