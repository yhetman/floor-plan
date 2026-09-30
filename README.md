# 3D floor plan to 2D room layout

Exploratory prototype that turns one 3D apartment render into an approximate 2D room layout. It outlines visible room-like regions, simplifies their boundaries to polygons, and reports each region's share of the detected floor area in pixels.

## Pipeline

Classical computer vision only. No learned model. `python -m src.main` reads each image, calls `extract_layout` in `src/pipeline.py`, and writes one PNG and one JSON file.

Lengths below are fractions of the shorter image side. Coordinates stay in the source image: origin at the top left, x to the right, y down. The drawing uses that same frame, so a JSON vertex is a pixel on the annotated image.

```
image
  │
  ├─ apartment mask          white page removed, holes filled
  ├─ wall segments           long, axis-aligned, thick, bright edges
  ├─ merged walls + doors    collinear faces joined, door gaps recorded
  ├─ structural filter       lines that never touch the shell are dropped
  ├─ rooms                   components left inside the footprint
  ├─ openings                windows, balcony rails, door leaves
  └─ schematic               orthogonal plan, symbols, pixel areas
```

1. **Apartment mask** (`_apartment_mask`). Near-white pixels are the page. The largest remaining blob is the plan. Holes are filled so a caption under the drawing is not part of the floor. The same step stores lightness and chroma; later stages use those to tell wall paint from floor and furniture.

2. **Wall segments** (`_wall_segments`). A line-segment detector proposes edges. An edge is kept when it is within about 12 degrees of horizontal or vertical, lies mostly inside the plan, is bright and low-chroma, and has a thick bright band across it. A thin furniture edge or a dark sofa outline fails that last check. Each kept edge is stored as `(transverse, start, end)`: y and the x-interval for a horizontal wall, x and the y-interval for a vertical one.

3. **Merge and bridge** (`_merge_collinear`). The two faces of one wall sit a fraction of the image apart. Faces within that distance are snapped onto one line. A gap along that line up to about 12% of the short side is bridged, so a doorway does not join two rooms. Gaps wider than about 3.5% are remembered as door candidates.

4. **Drop furniture** (`_keep_structural`). Wall pieces are drawn into a barrier and connected. A component that never touches the outer contour of the plan is removed. That is what keeps a closet rod or a counter from becoming a wall.

5. **Rooms** (`_rooms_from_barrier`). The barrier, plus the outer contour, is subtracted from the plan. Connected components that are large enough become rooms. Thin, very bright strips are the top of an isometric wall and are discarded. Each mask is simplified to a polygon. `area_px` is the mask's pixel count. `relative_area` is that count divided by the sum of the room counts.

6. **Openings** (`src/openings.py`). Windows and balcony rails share one line-segment pass. Doors are the gaps remembered in step 3. A mark that fails its check is not drawn.
   - **Window.** Two bright frame lines on the exterior, with a bright, pale, textured strip between them. The strip has to be long enough to be glazing. A flat gray wall face, a picture frame, and a bathtub rim fail this.
   - **Balcony.** Either a thin dark component on the exterior (a metal rail) or a pale exterior line with at least five baluster slats just inside it. A span that is already a window is not also a rail.
   - **Door.** A remembered gap whose width is between about 3% and 12% of the short side, and whose pixels are not still uniform wall paint.

7. **Schematic** (`src/schematic.py`). The footprint is squared into an orthogonal shell, and interior walls are snapped so each end stops on the wall it meets. The shell is a thick black band. Interior walls are thinner black bars. A door gap is cut out of its wall and a quarter-circle swing is drawn into the room. A window becomes three thin lines in that stretch of the exterior band. A balcony rail replaces the thick band with two thin lines. Each room is labeled with its pixel count and its share of the detected floor. Room names, furniture, and feet-and-inch sizes are not drawn.

Outputs for each input image:

- `*_annotated.png` — the 2D plan.
- `*.json` — polygons, pixel areas, relative areas, and the drawn window, door, and balcony spans.

## Assumptions

- The picture is a top-down or mildly axonometric marketing render, not a perspective photo.
- Walls are mostly axis-aligned in the image, brighter, and less saturated than the floors.
- Door and cased openings are narrower than about 12% of the shorter image side. Wider openings (open kitchens, balcony sliders) stay connected.
- Occlusion is limited: furniture may cover the floor, but it is not treated as a wall unless it looks like a long bright edge.
- Areas are counts of room pixels in the image. Axonometric wall height stretches some regions. Furniture inside a room is ignored.

Manual preprocessing is not required for the samples in `input_examples/`.

## Limitations

- Open-plan living and kitchen areas often stay one region when the opening is wider than a door.
- A real door is missed when the wall edges on either side are not collinear enough to snap together, so those rooms merge.
- Counters, closet rods, and window mullions sometimes survive as extra barriers and split a room.
- Balconies behind a railing are easy to miss or to merge into the next room.
- The polygon is an axis-aligned approximation of the visible floor, including some isometric wall face. It is not a surveyed plan.
- Very light tile can look like wall paint, so bathrooms are the least stable rooms.

## Run with Docker

From the repository root:

```bash
docker compose up --build
```

Annotated images and JSON are written to `./output`. To process a different folder, change the volume mount and keep the container paths `/data/input` and `/data/output`.

One file:

```bash
docker compose run --rm floorplan --input /data/input/your_plan.webp --output-dir /data/output
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.main --input-dir input_examples --output-dir output
```

## JSON shape

```json
{
  "source_image": "plan.webp",
  "image_width": 1140,
  "image_height": 855,
  "coordinate_system": "origin at the top-left corner, x to the right, y down",
  "area_unit": "pixels",
  "plan_area_px": 200000,
  "total_room_area_px": 120000,
  "features": [
    {"kind": "window", "orientation": "horizontal", "x0": 351, "y0": 48, "x1": 483, "y1": 48},
    {"kind": "door", "orientation": "vertical", "x0": 597, "y0": 323, "x1": 597, "y1": 421},
    {"kind": "balcony", "orientation": "horizontal", "x0": 597, "y0": 110, "x1": 782, "y1": 110}
  ],
  "rooms": [
    {
      "id": "room_1",
      "polygon": [[x, y], [x, y]],
      "area_px": 50000,
      "relative_area": 0.4167,
      "centroid": [400.0, 300.0],
      "bbox": [x, y, width, height]
    }
  ]
}
```

`polygon` is an ordered list of pixel vertices. The ring is closed implicitly (the last vertex connects back to the first). `bbox` is `[x, y, width, height]`. Each item in `features` is a detected window, door, or balcony. Its line is the wall span where that symbol is drawn.

## Next steps

- Replace the hand-tuned wall thresholds with a small segmentation model trained on axonometric plans, and keep this pipeline as a weak labeler.
- Close doors by matching wall endpoints in a graph instead of a single collinear gap, so skewed isometric walls still join.
- Split open-plan areas only when floor material changes across a long, straight boundary, and ignore rugs.
- Report area in image pixels only. A later step can calibrate those counts to printed dimensions when a render includes them.

If an annotated set was available (room polygons, wall lines, and window, door, and balcony spans on more renders than the three samples):

- Score each stage on a held-out split: room polygon overlap, error in `relative_area`, and how often a drawn window, door, or balcony lands on a labeled span. The cutoffs in `src/pipeline.py` and `src/openings.py` (wall thickness, door width, glass gap, edge texture, baluster count) would be chosen by that score instead of by the three examples.
- Train a wall / floor / opening segmenter on those labels. The classical stages can stay as a check where the model is uncertain, and as a way to draft labels for renders that are not annotated yet.
- Label furniture and fixtures as their own class, so a closet rod or a counter is rejected because it was labeled that way, rather than because it fails to touch the outer shell.
- Where a label includes the printed square footage, fit a scale from `area_px` to floor area for that render style. Pixel counts stay the measurement; the scale is a second field.
