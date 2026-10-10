#!/usr/bin/env python3
"""Lift the exact L55 Metal arithmetic to the traced 512x288 shape."""

from __future__ import annotations

import argparse
from pathlib import Path


BEFORE = """  if (p.physical_width != 640u || p.physical_height != 368u ||
      p.accumulation_mode != 1u || p.candidate_width != 320u ||
      p.candidate_height != 180u || p.secondary_width != 320u ||
      p.secondary_height != 180u || p.active_width != 640u ||
      p.active_height != 360u || p.optional_ui_enabled != 0u ||
      position.x >= p.physical_width || position.y >= p.physical_height) {
"""
AFTER = """  if (p.accumulation_mode != 1u ||
      p.candidate_width != p.secondary_width ||
      p.candidate_height != p.secondary_height ||
      p.active_width > p.physical_width ||
      p.active_height > p.physical_height || p.optional_ui_enabled != 0u ||
      position.x >= p.physical_width || position.y >= p.physical_height) {
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"preserve existing output: {args.output}")
    source = args.source.read_text()
    if source.count(BEFORE) != 1:
        raise RuntimeError("L55 selected-shape guard anchor drift")
    result = (
        "// GENERATED exact dynamic-shape L55 specialization; do not edit.\n"
        + source.replace(BEFORE, AFTER)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result)
    print(f"generated={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
