#!/usr/bin/env python3
"""Build Netal's native Metal DLSS 5 Neural Rendering library, metallib, and CLI."""
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
    out = subprocess.check_output(['xcrun', '--find', name], text=True).strip()
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

    header_src = ROOT / 'src/nr_fast/nr_fast_runtime.h'
    runtime_src = ROOT / 'src/nr_fast/nr_fast_runtime.mm'
    kernels_src = ROOT / 'src/nr_fast/nr_fast_kernels.metal'
    cli_src = ROOT / 'src/nr_fast/nr_fast_cli.mm'

    for src in (header_src, runtime_src, kernels_src, cli_src):
        if not src.is_file():
            raise FileNotFoundError(f'Missing source file: {src}')

    metal_flags = [
        '-std=metal3.1',
        '-fmetal-math-mode=safe',
        '-O3',
        '-Wall',
        '-Wextra',
    ]
    cxx_flags = [
        '-std=c++20',
        '-arch',
        'arm64',
        '-fobjc-arc',
        '-O3',
        '-Wall',
        '-Wextra',
        f'-I{header_src.parent}',
    ]
    frameworks = [
        '-framework', 'Foundation',
        '-framework', 'Metal',
        '-framework', 'CoreGraphics',
        '-framework', 'ImageIO',
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.nr-fast-build-', dir=output.parent) as tmp:
        tmp_dir = Path(tmp)
        stage_dir = tmp_dir / 'artifacts'
        stage_dir.mkdir()

        air_path = tmp_dir / 'nr_fast_kernels.air'
        metallib_path = stage_dir / 'nr_fast_kernels.metallib'
        runtime_obj = tmp_dir / 'nr_fast_runtime.o'
        static_lib = stage_dir / 'libnetal_nr_fast.a'
        cli_bin = stage_dir / 'netal-nr-fast-cli'

        subprocess.run(
            [str(metal_cc), *metal_flags, '-c', str(kernels_src), '-o', str(air_path)],
            check=True,
        )
        subprocess.run(
            [str(metallib_ld), str(air_path), '-o', str(metallib_path)],
            check=True,
        )
        subprocess.run(
            [str(clangxx), *cxx_flags, '-c', str(runtime_src), '-o', str(runtime_obj)],
            check=True,
        )
        subprocess.run(
            [str(libtool), '-static', '-o', str(static_lib), str(runtime_obj)],
            check=True,
        )
        subprocess.run(
            [
                str(clangxx),
                *cxx_flags,
                str(cli_src),
                str(static_lib),
                *frameworks,
                '-o',
                str(cli_bin),
            ],
            check=True,
        )

        receipt = {
            'schema': 'netal.nr.fast-build.v1',
            'metal_compiler': subprocess.check_output([str(metal_cc), '--version'], text=True).strip(),
            'cxx_compiler': subprocess.check_output([str(clangxx), '--version'], text=True).strip(),
            'sources': [
                {
                    'path': str(s.relative_to(ROOT)),
                    'sha256': sha256_file(s),
                }
                for s in (header_src, runtime_src, kernels_src, cli_src)
            ],
            'artifacts': [
                {
                    'name': p.name,
                    'bytes': p.stat().st_size,
                    'sha256': sha256_file(p),
                }
                for p in (metallib_path, static_lib, cli_bin)
            ],
            'metal_flags': metal_flags,
            'cxx_flags': [f for f in cxx_flags if not f.startswith('-I')],
        }
        (stage_dir / 'build-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        stage_dir.rename(output)

    print(f'result=pass artifacts=3 output={output.name}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
