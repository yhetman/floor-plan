"""Run the floor-plan pipeline on one image or a directory of images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from src.pipeline import extract_layout

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Approximate 2D room layouts from 3D floor-plan images.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="One floor-plan image")
    source.add_argument("--input-dir", type=Path, help="Directory of floor-plan images")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for annotated images and JSON (default: ./output)",
    )
    args = parser.parse_args(argv)

    if args.input is None and args.input_dir is None:
        default_dir = Path("input_examples")
        if not default_dir.is_dir():
            parser.error("pass --input or --input-dir")
        args.input_dir = default_dir

    images = _collect_images(args.input, args.input_dir)
    if not images:
        print("No images found.", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for path in images:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Could not read {path}", file=sys.stderr)
            failures += 1
            continue
        annotated, result = extract_layout(image, path.name)
        stem = path.stem.replace(" ", "_")
        image_path = args.output_dir / f"{stem}_annotated.png"
        json_path = args.output_dir / f"{stem}.json"
        cv2.imwrite(str(image_path), annotated)
        json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"{path.name}: {len(result['rooms'])} rooms -> {image_path.name}, {json_path.name}")

    return 1 if failures else 0


def _collect_images(single: Path | None, directory: Path | None) -> list[Path]:
    if single is not None:
        return [single]
    assert directory is not None
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )


if __name__ == "__main__":
    raise SystemExit(main())
