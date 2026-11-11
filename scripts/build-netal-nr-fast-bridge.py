#!/usr/bin/env python3
# SPDX-License-Identifier: MIT AND Apache-2.0
"""Build Netal's opt-in continuous in-process DLSS 5 NR bridge dylib and metallibs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_tool(name: str) -> Path:
    out = subprocess.check_output(['xcrun', '--sdk', 'macosx', '--find', name], text=True).strip()
    return Path(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Destination directory (must not already exist; populated atomically).',
    )
    args = parser.parse_args()
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        parser.error(f'Output already exists: {output}')

    metal_cc = find_tool('metal')
    metallib_ld = metal_cc.with_name('metallib')
    if not metallib_ld.is_file():
        metallib_ld = find_tool('metallib')
    clangxx = find_tool('clang++')
    libtool = find_tool('libtool')

    nr_header = ROOT / 'src/nr_fast/nr_fast_runtime.h'
    nr_runtime = ROOT / 'src/nr_fast/nr_fast_runtime.mm'
    nr_kernels = ROOT / 'src/nr_fast/nr_fast_kernels.metal'
    flow_header = ROOT / 'src/macos_host/netal_nr_optical_flow.h'
    flow_kernels = ROOT / 'src/macos_host/netal_nr_optical_flow.metal'
    state_header = ROOT / 'src/macos_host/netal_nr_continuous_state.h'
    bridge_src = ROOT / 'src/macos_host/netal_nr_fast_bridge.mm'

    sources = (
        nr_header,
        nr_runtime,
        nr_kernels,
        flow_header,
        flow_kernels,
        state_header,
        bridge_src,
    )
    for src in sources:
        if not src.is_file():
            raise FileNotFoundError(f'Missing source file: {src}')

    metal_fast_flags = [
        '-std=metal3.1',
        '-fmetal-math-mode=safe',
        '-O3',
        '-Wall',
        '-Wextra',
    ]
    metal_flow_flags = [
        '-std=metal3.1',
        '-O2',
        '-Wall',
        '-Wextra',
        '-Werror',
    ]
    cxx_flags = [
        '-std=c++20',
        '-arch',
        'arm64',
        '-fobjc-arc',
        '-O3',
        '-Wall',
        '-Wextra',
        '-Werror',
        f'-I{nr_header.parent}',
        f'-I{bridge_src.parent}',
    ]
    bridge_frameworks = [
        '-framework', 'Foundation',
        '-framework', 'Metal',
        '-framework', 'MetalFX',
        '-framework', 'QuartzCore',
        '-framework', 'CoreGraphics',
        '-framework', 'ImageIO',
        '-framework', 'UniformTypeIdentifiers',
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.nr-fast-bridge-build-', dir=output.parent) as tmp:
        tmp_dir = Path(tmp)
        stage_dir = tmp_dir / 'artifacts'
        stage_dir.mkdir()

        nr_air = tmp_dir / 'nr_fast_kernels.air'
        nr_metallib = stage_dir / 'nr_fast_kernels.metallib'
        flow_air = tmp_dir / 'netal_nr_optical_flow.air'
        flow_metallib = stage_dir / 'netal_nr_optical_flow.metallib'
        nr_obj = tmp_dir / 'nr_fast_runtime.o'
        static_lib = stage_dir / 'libnetal_nr_fast.a'
        bridge_dylib = stage_dir / 'libNetalNRFastBridge.dylib'

        subprocess.run(
            [str(metal_cc), *metal_fast_flags, '-c', str(nr_kernels), '-o', str(nr_air)],
            check=True,
        )
        subprocess.run(
            [str(metallib_ld), str(nr_air), '-o', str(nr_metallib)],
            check=True,
        )
        subprocess.run(
            [str(metal_cc), *metal_flow_flags, '-c', str(flow_kernels), '-o', str(flow_air)],
            check=True,
        )
        subprocess.run(
            [str(metallib_ld), str(flow_air), '-o', str(flow_metallib)],
            check=True,
        )
        subprocess.run(
            [str(clangxx), *cxx_flags, '-c', str(nr_runtime), '-o', str(nr_obj)],
            check=True,
        )
        subprocess.run(
            [str(libtool), '-static', '-o', str(static_lib), str(nr_obj)],
            check=True,
        )
        subprocess.run(
            [
                str(clangxx),
                *cxx_flags,
                '-dynamiclib',
                '-Wl,-install_name,@rpath/libNetalNRFastBridge.dylib',
                str(bridge_src),
                str(static_lib),
                *bridge_frameworks,
                '-o',
                str(bridge_dylib),
            ],
            check=True,
        )
        subprocess.run(
            ['codesign', '--force', '--sign', '-', str(bridge_dylib)],
            stdout=subprocess.DEVNULL,
            check=True,
        )

        artifacts = (nr_metallib, flow_metallib, static_lib, bridge_dylib)
        receipt = {
            'schema': 'netal.nr.fast-bridge-build.v1',
            'metal_compiler': subprocess.check_output([str(metal_cc), '--version'], text=True).strip(),
            'cxx_compiler': subprocess.check_output([str(clangxx), '--version'], text=True).strip(),
            'sources': [
                {
                    'path': str(s.relative_to(ROOT)),
                    'sha256': sha256_file(s),
                }
                for s in sources
            ],
            'artifacts': [
                {
                    'name': p.name,
                    'bytes': p.stat().st_size,
                    'sha256': sha256_file(p),
                }
                for p in artifacts
            ],
            'metal_fast_flags': metal_fast_flags,
            'metal_flow_flags': metal_flow_flags,
            'cxx_flags': [f for f in cxx_flags if not f.startswith('-I')],
        }
        (stage_dir / 'build-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
        stage_dir.rename(output)

    print(f'result=pass artifacts=4 output={output.name}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
