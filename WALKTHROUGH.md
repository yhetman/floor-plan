# Walkthrough

How to run the prototype, where the code lives, what happens to one render, and what the three sample plans actually show.

The short version of the method, assumptions, and limitations is in [README.md](README.md).

## Setup

Python 3.11. The only libraries are NumPy, OpenCV (the 4.9 line, headless), and Pillow. OpenCV 5 is pinned out: its line-segment detector returns a different array shape, and the wall and opening code unpacks the 4.x layout.

Docker is the intended way to run it. From the repository root:

```bash
docker compose up --build
```

That builds `floorplan-layout` from the Dockerfile, mounts `input_examples/` read-only at `/data/input`, mounts `output/` at `/data/output`, and runs every image in the input folder. Each image produces two files in `output/`:

- `{name}_annotated.png` — the plan, same pixel size as the source.
- `{name}.json` — polygons, pixel areas, and the window, door, and balcony spans that were drawn.

The image installs `libglib2.0-0` because headless OpenCV needs it, and Liberation Sans because the area labels are drawn with that face. There are no model weights.

One file, still inside the container paths:

```bash
docker compose run --rm floorplan --input /data/input/your_plan.webp --output-dir /data/output
```

Locally, the same entry point:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.main --input-dir input_examples --output-dir output
```

`--input` takes a single image. With no arguments, `src.main` reads `input_examples/`. Accepted suffixes are png, jpg, jpeg, webp, bmp, tif, and tiff.

## Code structure

```
src/main.py        CLI: read images, write PNG + JSON
src/pipeline.py    walls, rooms, and the order of stages
src/openings.py    windows, balcony rails, door-gap filter
src/schematic.py   orthogonal drawing and area labels
```

`main.py` does not interpret the plan. It loads a BGR image, calls `extract_layout`, and writes the two output files. The printed line is the room count and the output names.

`pipeline.py` owns detection of the apartment and of the walls. `extract_layout` is the only public function. It returns the annotated image and the JSON dict. Wall pieces are stored as `(transverse, start, end)`: for a horizontal wall, y and the x-interval; for a vertical wall, x and the y-interval. Length cutoffs are fractions of the shorter image side, so the same constants apply to the 659×651 render and the 1140×855 ones.

`openings.py` owns everything that is not a wall. `detect_openings` returns windows and balconies. `filter_doors` takes the gaps that the wall merger already found and drops the ones that are still solid paint. A segment here is `(orientation, transverse, start, end)` with orientation `"h"` or `"v"`.

`schematic.py` owns the drawing. `render_schematic` takes the wall lists, door gaps, room masks, and opening segments, and returns the canvas plus the feature spans that were actually snapped onto a wall. A detection that cannot be placed on the shell is left out of both the picture and the JSON.

## Processing pipeline

Coordinates never leave the source image. The origin is the top left, x increases to the right, y increases downward. A vertex in the JSON is a pixel on the annotated PNG.

1. **Cut out the apartment.** Near-white pixels are the page. The largest remaining blob is the plan. Holes inside it are filled, so a caption under the drawing is not floor. Lightness and chroma are kept for the later tests that separate wall paint from floor and furniture.

2. **Keep wall edges.** A line-segment detector proposes edges on a lightly blurred gray image. An edge is kept when it is within about 12 degrees of horizontal or vertical, sits inside the plan, is bright and low-chroma, and has a thick bright band across it. A thin furniture edge or a dark sofa outline fails that last check.

3. **Join the two faces of each wall.** Parallel faces within a wall-thickness of each other snap onto one line. A gap along that line, up to about 12% of the short side, is bridged so a doorway does not merge two rooms. Gaps wider than about 3.5% are remembered as door candidates.

4. **Drop furniture.** The remaining pieces are drawn into a barrier. A connected piece that never touches the outer contour of the plan is removed. That is what keeps a closet rod or a counter from becoming a wall.

5. **Label rooms.** The barrier and the outer contour are subtracted from the plan. Connected components that are large enough become rooms. Thin, very bright strips are the top of an isometric wall and are discarded. Each mask is simplified to a polygon. `area_px` is the mask's pixel count. `relative_area` is that count divided by the sum of the room counts, not by the whole image and not by the printed square footage in the filename.

6. **Find openings.** Windows and rails share one line-segment pass. A window is two bright frame lines on the exterior with a bright, pale, textured strip between them. A balcony is a thin dark rail on the exterior, or a pale exterior line with at least five baluster slats just inside it. A span that is already a window is not also a rail. A door is a remembered gap of about 3–12% of the short side whose pixels are not still uniform wall paint. A mark that fails its check is not drawn.

7. **Draw the plan.** The footprint is squared into an orthogonal shell. Interior walls snap so each end stops on the wall it meets. The shell is a thick black band; interior walls are thinner black bars. A door is a gap in that bar plus a quarter-circle swing into the room that has floor on that side. A window is three thin lines in the exterior band. A balcony rail replaces the thick band with two thin lines along that span. Each room is labeled with its pixel count and its share of the detected floor.

## Comments on the results

The three samples are isometric marketing renders. The drawings recover the outer shell and the main partitions. They are not surveyed plans: axonometric wall faces add pixels, furniture can still split a room, and an opening wider than a door leaves two spaces as one region. Pixel shares below are each room's share of detected room pixels.

**Heritage Towers A1** (1140×855, filename 593 sq ft) → 4 rooms.

| Room | Pixels | Share |
| --- | ---: | ---: |
| room_1 | 180531 | 58% |
| room_2 | 73745 | 24% |
| room_3 | 34154 | 11% |
| room_4 | 21454 | 7% |

The large left region is one room because the living area is open; the detector does not invent a kitchen line that is not a wall. The top edge of that room is drawn as a window (three lines, span x=351–483). The notch between the 58% and 24% rooms is drawn as a balcony rail (two lines, x=597–782). One door swing sits on the wall between the 58% room and the 7% room. Other glazing on this render did not pass the frame-and-glass checks, so those walls stay solid. The 11% and 7% rooms are the small enclosed spaces on the right and bottom; their shares are plausible for a bath and a closet, and the labels are pixel shares, not the 593 sq ft in the filename.

**Highland Lux Citadel** (659×651, filename 623 sq ft) → 8 rooms.

| Room | Pixels | Share |
| --- | ---: | ---: |
| room_1 | 67242 | 29% |
| room_2 | 43741 | 19% |
| room_3 | 41427 | 18% |
| room_4 | 21733 | 9% |
| room_5 | 21280 | 9% |
| room_6 | 18812 | 8% |
| room_7 | 11929 | 5% |
| room_8 | 7408 | 3% |

This plan is split more finely than a one-bedroom marketing plan. The 29%, 19%, and 18% regions are the main rooms, and the walls between them meet at corners. Three doors are drawn: one into the top-left pair of rooms, one between the 29% and 9% rooms, and a short swing on the right side of the 19% room. The bottom-right rail is a balcony (two lines, x=394–527). The step just outside that rail stays a solid wall; the baluster test did not fire there. No window was drawn. The 3% region is a real connected component, left as detected: it is small enough to be a closet or a pocket next to a door, and it is also small enough that a fixture edge may have created it. The 5% and 8% rooms on the top left are separate because a wall was found between them.

**Limestone Ranch, Santa Fe** (1140×855, filename 625 sq ft) → 7 rooms.

| Room | Pixels | Share |
| --- | ---: | ---: |
| room_1 | 85958 | 33% |
| room_2 | 78718 | 31% |
| room_3 | 31837 | 12% |
| room_4 | 31750 | 12% |
| room_5 | 11347 | 4% |
| room_6 | 9024 | 4% |
| room_7 | 8456 | 3% |

The outline keeps the left wing. That wing is two narrow 4% regions, one above the other, rather than a single room. The 33% room on the right has a window along its top wall (x=663–786). The notch at the top of the 31% room is a balcony rail (x=440–579). Three doors are drawn on the partitions around the 31% room: one horizontal leaf and two vertical leaves stacked on the same wall, which is why two swings sit close together. The 3% cell under the left wing is the least trustworthy region on this plan; a short interior edge can close a box that is not a separate room. The two 12% rooms along the bottom are clean rectangles and match the partitions in the render.

Across all three, `plan_area_px` is larger than the sum of `area_px` because walls, and the thin isometric wall tops that were discarded, are not rooms. Relative area will not match the square footage printed on the filename until a scale from pixels to floor area is fitted, and that scale is not in this prototype.
