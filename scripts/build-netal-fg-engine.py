#!/usr/bin/env python3
"""Compile FG engine libraries and host archives without models or inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tool_version(command):
    # Retain the version, without machine-specific InstalledDir lines.
    return subprocess.check_output(command, text=True).splitlines()[0]


def build(output):
    recipe = json.loads((ROOT / 'src/fg_runtime/kernel-build.json').read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    compiler = Path(subprocess.check_output(['xcrun', '--find', 'metal'], text=True).strip())
    linker = compiler.with_name('metallib')
    if not linker.is_file():
        linker = Path(subprocess.check_output(['xcrun', '--find', 'metallib'], text=True).strip())

    with tempfile.TemporaryDirectory(prefix='.fg-engine-', dir=output.parent) as temporary:
        temporary = Path(temporary)
        stage = temporary / 'engine'
        libs = stage / 'libs'
        generated = stage / 'generated'
        overlay = generated / 'overlays'
        libdir = stage / 'lib'
        include = stage / 'include'
        for directory in (libs, overlay, libdir, include):
            directory.mkdir(parents=True)
        subprocess.run([
            'python3', str(ROOT / 'scripts/generate-netal-fg-overlays.py'),
            str(ROOT), str(overlay), '1',
        ], check=True)
        subprocess.run([
            'python3', str(ROOT / 'scripts/generate-netal-fg-tail.py'),
            str(ROOT / 'src/fg_connected_l20_l87/macos_connected_l20_l87.mm'),
            str(generated / 'runtime_tail.mm'),
            str(ROOT / 'src/fg_runtime/tail_resource_cache.inc'),
        ], check=True)
        overrides = {
            '$conv3_source': overlay / 'conv_3x3_live_zero.metal',
            '$l52_source': overlay / 'l52_compact.metal',
            '$l54_source': overlay / 'l54_compact.metal',
            '$bot0_source': overlay / 'block0_bot0_live_zero.metal',
        }
        records = []

        def compile_kernel(source, name, standard, override=None):
            air = temporary / (name + '.air')
            target = libs / (name + '.metallib')
            flags = [
                '-std=' + standard, '-fno-fast-math', '-ffp-contract=off',
                '-I', str(ROOT / 'src/fg_conv'), '-I', str(ROOT / 'src/fg_math_ex2'),
            ]
            if override:
                flags += ['-include', str(override)]
            subprocess.run([str(compiler), *flags, '-c', str(source), '-o', str(air)], check=True)
            subprocess.run([str(linker), str(air), '-o', str(target)], check=True)
            source_path = source.relative_to(stage if source.is_relative_to(stage) else ROOT)
            records.append({
                'library': target.name, 'source': str(source_path),
                'source_sha256': sha(source),
                'override': str(override.relative_to(ROOT)) if override else None,
                'sha256': sha(target),
            })

        for item in recipe['kernels']:
            source = overrides.get(item['source'], ROOT / item['source'])
            name, standard = item['name'], item['metal_standard']
            hmma = 'hmma_exact.metalh' in source.read_text()
            if hmma:
                hardware = ROOT / f'src/fg_conv/{name}_hw.metal'
                compile_kernel(
                    hardware if hardware.exists() else source, name + '_hw', standard,
                    None if hardware.exists() else ROOT / 'src/fg_conv_simd_l20/hmma_hw_override.metalh',
                )
            compile_kernel(
                source, name, standard,
                ROOT / 'src/fg_conv_simd_l20/hmma_exact_u32_override.metalh' if hmma else None,
            )
            print('compiled=' + name, flush=True)

        sources = [ROOT / 'src/fg_runtime' / name for name in (
            'fg_runtime.mm', 'metal4_shim.mm', 'span_recorder.mm', 'tail_output_bridge.mm',
        )] + [
            ROOT / 'src/fg_production_contract_l3_l87/macos_live_handoff.mm',
            generated / 'runtime_tail.mm',
        ]
        for arch in ('x86_64', 'arm64'):
            objects = []
            for source in sources:
                obj = temporary / (arch + '-' + source.stem + '.o')
                objects.append(obj)
                subprocess.run([
                    'xcrun', 'clang++', '-arch', arch, '-std=c++20', '-fobjc-arc',
                    '-ffp-contract=off', '-O2', '-mmacosx-version-min=26.0',
                    '-Wall', '-Wextra', '-Wno-unused-function', '-Wno-deprecated-declarations',
                    '-I', str(ROOT / 'src/fg_runtime'),
                    '-I', str(ROOT / 'src/fg_production_contract_l3_l87'),
                    '-I', str(ROOT / 'src/fg_connected_l20_l87'),
                    '-c', str(source), '-o', str(obj),
                ], check=True)
            subprocess.run([
                'xcrun', 'libtool', '-static', '-o', str(libdir / f'libnetal_fg_runtime-{arch}.a'),
                *[str(p) for p in objects],
            ], check=True, env=dict(os.environ, ZERO_AR_DATE='1'))
        subprocess.run([
            'xcrun', 'lipo', '-create', str(libdir / 'libnetal_fg_runtime-x86_64.a'),
            str(libdir / 'libnetal_fg_runtime-arm64.a'),
            '-o', str(libdir / 'libnetal_fg_runtime.a'),
        ], check=True)
        shutil.copy2(ROOT / 'src/fg_runtime/fg_runtime.h', include / 'fg_runtime.h')
        receipt = {
            'schema': 'netal.fg.source-engine.v1',
            'toolchain': {
                'metal': tool_version([str(compiler), '--version']),
                'clang': tool_version(['xcrun', 'clang++', '--version']),
                'sdk': tool_version(['xcrun', '--show-sdk-version']),
            },
            'profile': '640x360', 'hmma': 'u32', 'ex2': 'compact', 'live_zero': True,
            'kernels': records, 'generated_tail_sha256': sha(generated / 'runtime_tail.mm'),
            'model_inputs': 0, 'gpu_executed': False, 'complete_runtime': False,
            'gameplay_validated': False,
            'files': {
                p.relative_to(stage).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)}
                for p in sorted(stage.rglob('*')) if p.is_file()
            },
        }
        (stage / 'build-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        # Recheck after compilation, so a concurrent completed build is preserved.
        if output.exists() or output.is_symlink():
            raise FileExistsError(f'Output exists: {output}')
        stage.rename(output)
    print(f'result=pass libraries={len(records)} model_inputs=0 gpu_executed=false complete_runtime=false')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        parser.error('Output exists')
    build(output)


if __name__ == '__main__':
    main()
