#!/usr/bin/env python3
"""Build the native iOS SR static library and explicitly targeted Metal kernels.

This source-only build does not supply a model, scalar contracts, an app signature,
RTX correctness evidence, or an iPhone performance result. It omits the macOS app,
CrossOver/native-game interposer, FG engine and NVIDIA binaries.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from netal_apple_build import check_macho_platform, metal_target, version_tuple

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    'src/macos_host/dlss_sr_session.mm',
    'src/macos_host/dlss_sr_texture_adapter.mm',
    'src/ios_host/netal_ios_sr_device_policy.cpp',
    'src/ios_host/netal_ios_sr_capabilities.mm',
)
HEADERS = (
    'src/ios_host/NetalIOSSR.h',
    'src/ios_host/netal_ios_sr_capabilities.h',
    'src/macos_host/dlss_sr_session.h',
    'src/macos_host/dlss_sr_texture_adapter.h',
    'src/macos_host/dlss_sr_frame_ledger.h',
    'src/macos_host/dlss_sr_neural_backend.h',
)
KERNEL_BUILDERS = {
    'reference': 'build-netal-sr-reference-kernels.py',
    'boundary': 'build-netal-sr-boundary-kernels.py',
    'texture': 'build-netal-sr-texture-kernels.py',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(output, sdk='iphoneos', deployment_target='26.0', kernels='all'):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError('output exists; choose a new directory')
    if sdk != 'iphoneos':
        raise ValueError('the SR core requires the physical iphoneos SDK; the simulator SDK does not expose Metal 4')
    deployment_target, metal_triple = metal_target(sdk, deployment_target)
    if kernels not in ('none', 'texture', 'all'):
        raise ValueError('unsupported kernel selection')
    for relative in SOURCES + HEADERS:
        if not (ROOT / relative).is_file():
            raise ValueError('missing source dependency: ' + relative)
    sdk_root = subprocess.check_output(['xcrun', '--sdk', sdk, '--show-sdk-path'], text=True).strip()
    sdk_version = subprocess.check_output(['xcrun', '--sdk', sdk, '--show-sdk-version'], text=True).strip()
    if version_tuple(deployment_target) > version_tuple(sdk_version):
        raise ValueError('deployment target is newer than the installed SDK')
    clang = subprocess.check_output(['xcrun', '--sdk', sdk, '--find', 'clang++'], text=True).strip()
    libtool = subprocess.check_output(['xcrun', '--sdk', sdk, '--find', 'libtool'], text=True).strip()
    triple = f'arm64-apple-ios{deployment_target}'
    platform = 2
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.netal-ios-sr-', dir=output.parent) as temp:
        temp = Path(temp)
        stage = temp / 'output'
        (stage / 'lib').mkdir(parents=True)
        include = stage / 'include/NetalIOSSR'
        include.mkdir(parents=True)
        objects = []
        records = []
        for relative in SOURCES:
            source = ROOT / relative
            obj = temp / (source.stem + '.o')
            flags = ['-target', triple, '-isysroot', sdk_root, '-std=c++20', '-O2', '-Wall', '-Wextra']
            if source.suffix == '.mm':
                flags.append('-fobjc-arc')
            flags += ['-I', str(ROOT / 'src/macos_host'), '-I', str(ROOT / 'src/ios_host')]
            subprocess.run([clang, *flags, '-c', str(source), '-o', str(obj)], check=True)
            check_macho_platform(obj, platform, deployment_target)
            objects.append(obj)
            records.append({'source': relative, 'source_sha256': sha(source), 'object_platform': platform})
            print('compiled=' + source.name, flush=True)
        archive = stage / 'lib/libNetalIOSSR.a'
        subprocess.run([libtool, '-static', '-o', str(archive), *map(str, objects)], check=True)
        for relative in HEADERS:
            shutil.copy2(ROOT / relative, include / Path(relative).name)
        selected = list(KERNEL_BUILDERS) if kernels == 'all' else ['texture'] if kernels == 'texture' else []
        for name in selected:
            subprocess.run([sys.executable, str(ROOT / 'scripts' / KERNEL_BUILDERS[name]),
                            '--sdk', sdk, '--deployment-target', deployment_target,
                            '--output', str(stage / 'kernels' / name)], check=True)
        (stage / 'LINKING.txt').write_text(
            'Use include/ and lib/libNetalIOSSR.a with an iOS 26+ arm64 application.\n'
            'Link Foundation, Metal and libc++; -ObjC is recommended for a static Objective-C library.\n'
            'A successful build is not model, inference, quality, acceleration or latency validation.\n'
            'Supply a separately assembled iOS runtime bundle with the supported local model/scalar inputs.\n')
        receipt = {
            'schema': 'netal.ios.sr.source-build.v1',
            'sdk': sdk, 'sdk_version': sdk_version, 'deployment_target': deployment_target,
            'cpu_target': triple, 'metal_target': metal_triple,
            'architecture': 'arm64', 'sources': records,
            'library': 'lib/libNetalIOSSR.a', 'library_sha256': sha(archive),
            'kernel_sets': selected, 'model_inputs': 0,
            'inference_validated': False, 'gameplay_validated': False,
            'hardware_acceleration_validated': False, 'latency_validated': False,
            'compiled_input_rights_review_required': bool(set(selected) & {'reference', 'boundary'}),
        }
        (stage / 'build-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        stage.rename(output)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sdk', choices=('iphoneos',), default='iphoneos',
                        help='physical-device SDK; Metal 4 SR is unavailable in the simulator SDK')
    parser.add_argument('--deployment-target', default='26.0')
    parser.add_argument('--kernels', choices=('none', 'texture', 'all'), default='all',
                        help='all builds retained reference/boundary sources too; no model is included')
    args = parser.parse_args()
    try:
        result = build(args.output, args.sdk, args.deployment_target, args.kernels)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print('result=fail ' + str(error), file=sys.stderr)
        return 1
    print('result=pass target=' + result['cpu_target'] + ' inference_validated=false')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
