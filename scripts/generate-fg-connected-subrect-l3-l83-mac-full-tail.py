#!/usr/bin/env python3
"""Bind the accepted generated L18-L83 Metal tail to live allocation37."""

from __future__ import annotations

import argparse
from pathlib import Path


def once(source: str, before: str, after: str, label: str) -> str:
    count = source.count(before)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return source.replace(before, after)


def generate(source: str) -> str:
    source = once(
        source,
        '#include "../fg_connected_subrect_l3_l83/contract.h"',
        '#include "../fg_connected_subrect_l3_l83/contract.h"\n'
        '#include "../fg_connected_subrect_l3_l83/'
        'macos_subrect_live_handoff.h"',
        'handoff include',
    )
    source = once(
        source,
        'int main(int argc, const char* argv[]) {',
        'int FgSubrectTailMain(int argc, const char* argv[]) {',
        'tail main',
    )
    source = once(
        source,
        '      const auto prefix_input_before = PrefixInput();\n'
        '      id<MTLBuffer> prefix_input =\n'
        '          Guarded(device, &prefix_input_before, '
        'subrect::kModelInputBytes, 0);',
        '      id<MTLBuffer> prefix_input = FgSubrectMacLivePrefixInput();\n'
        '      if (prefix_input == nil)\n'
        '        throw std::runtime_error("live subrect allocation37 binding '
        'missing");\n'
        '      const auto prefix_input_before =\n'
        '          Payload(prefix_input, subrect::kModelInputBytes);',
        'live prefix input',
    )
    return '// Generated live-input subrect tail; do not edit.\n' + source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f'preserve existing output: {args.output}')
    result = generate(args.source.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result)
    print(f'generated={args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
