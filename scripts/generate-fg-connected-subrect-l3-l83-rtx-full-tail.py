#!/usr/bin/env python3
"""Rename the accepted generated subrect RTX tail for in-process connection."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f'preserve existing output: {args.output}')
    source = args.source.read_text()
    before = 'int main(int argc, char** argv) {'
    if source.count(before) != 1:
        raise RuntimeError('tail main anchor drift')
    source = source.replace(
        before, 'int FgSubrectTailMain(int argc, char** argv) {')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('// Generated connected subrect RTX tail.\n' + source)
    print(f'generated={args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
