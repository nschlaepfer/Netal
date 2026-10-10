#!/usr/bin/env python3
"""Bind the selected L18-L49 CUDA prefix to live D3D12 allocation37."""

from __future__ import annotations

import argparse
from pathlib import Path


def once(source: str, before: str, after: str, label: str) -> str:
    count = source.count(before)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return source.replace(before, after)


LIVE_STATE = r'''
thread_local CUdeviceptr g_subrect_live_prefix_input = 0;

std::vector<std::uint8_t> ReadSubrectLivePrefixInput(
    CUdeviceptr source, std::size_t bytes) {
  std::vector<std::uint8_t> result(bytes);
  Check(cuMemcpyDtoH(result.data(), source, bytes),
        "read live subrect allocation37");
  return result;
}
'''


def generate(source: str) -> str:
    source = once(
        source,
        '#include "rtx_subrect_bridges.h"',
        '#include "rtx_subrect_bridges.h"\n'
        '#include "rtx_subrect_live_handoff.h"',
        'live header',
    )
    source = once(
        source,
        '}  // namespace\n\n\nint RunFgSubrectPrefixL18L49',
        LIVE_STATE + '\n}  // namespace\n\n'
        'void FgSubrectSetRtxLivePrefixInput(std::uint64_t input_cuda_ptr) {\n'
        '  g_subrect_live_prefix_input = '
        'static_cast<CUdeviceptr>(input_cuda_ptr);\n'
        '}\n\nint RunFgSubrectPrefixL18L49',
        'live state',
    )
    source = once(
        source,
        '        l51_output_cuda_ptr == 0 || result == nullptr) {',
        '        g_subrect_live_prefix_input == 0 ||\n'
        '        l51_output_cuda_ptr == 0 || result == nullptr) {',
        'live validation',
    )
    source = once(
        source,
        '    const auto input_before = GenerateSubrectEncoderInput();',
        '    const auto input_before = ReadSubrectLivePrefixInput(\n'
        '        g_subrect_live_prefix_input,\n'
        '        fg_connected_subrect_l3_l83::kModelInputBytes);',
        'live input snapshot',
    )
    source = once(
        source,
        '    Check(cuMemcpyHtoD(input->payload(),input_before.data(),input_before.size()),\n'
        '          "copy prefix input");',
        '    Check(cuMemcpyDtoD(input->payload(),g_subrect_live_prefix_input,\n'
        '                       fg_connected_subrect_l3_l83::kModelInputBytes),\n'
        '          "live allocation37 to L18 device handoff");',
        'live device copy',
    )
    source = once(
        source,
        '    std::size_t mismatches = Mismatches(input_before, input_after);',
        '    const auto live_input_after = ReadSubrectLivePrefixInput(\n'
        '        g_subrect_live_prefix_input,\n'
        '        fg_connected_subrect_l3_l83::kModelInputBytes);\n'
        '    std::size_t mismatches = Mismatches(input_before, input_after) +\n'
        '                             Mismatches(input_before, live_input_after);',
        'live immutability',
    )
    return '// Generated live allocation37 subrect CUDA prefix.\n' + source


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
