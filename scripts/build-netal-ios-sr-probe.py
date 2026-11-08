#!/usr/bin/env python3
"""Build an unsigned physical-iPhone UIKit SR probe; caller supplies runtime and signing.

The library root is the output of build-netal-ios-sr.py. No model, game data,
device identifier, signing identity or provisioning profile is built in.
--prepare-xcode also makes a self-contained project for caller-managed signing.
Optional benchmark.json render_size (for example 640x360) selects input dimensions
independently of the output profile; the device receipt records the actual ratio.
--fp8-lab-root optionally links the separate model-free operator lab; it requires
physical iPhone SDK 27+ and an explicit --deployment-target 27.0 or newer.
--metalfx-lab-root links the model-free system-scaler comparison. Captures require
a separate --enable-metal-capture build; ordinary timing builds keep capture disabled.
"""
import argparse
import hashlib
import json
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from netal_apple_build import check_macho_platform, version_tuple

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('NetalSRProbe.mm', 'NetalProbeMetrics.h', 'NetalProbeFP8Audit.h')


def validate(library_root, output, sdk, deployment_target, bundle_id, team_id=None):
    library_root = Path(library_root).absolute()
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError('output exists; choose a new directory')
    if sdk != 'iphoneos':
        raise ValueError('this probe requires the physical iphoneos SDK; simulator Metal 4 is unsupported')
    if version_tuple(deployment_target) < (26, 0, 0):
        raise ValueError('the SR core requires iOS 26.0 or newer')
    if not re.fullmatch(r'[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+', bundle_id):
        raise ValueError('invalid bundle identifier')
    if team_id and not re.fullmatch(r'[A-Z0-9]{10}', team_id):
        raise ValueError('team ID must be a caller-supplied ten-character identifier')
    for relative in ('lib/libNetalIOSSR.a', 'include/NetalIOSSR/NetalIOSSR.h'):
        if not (library_root / relative).is_file():
            raise ValueError('missing library input: ' + relative)
    return library_root, output


def load_fp8_lab(root, deployment_target):
    if version_tuple(deployment_target) < (27, 0, 0):
        raise ValueError('the optional operator lab requires --deployment-target 27.0 or newer')
    root = Path(root).absolute()
    for name in ('libnetal_ios_fp8_lab.a', 'NetalFP8MatmulLab.h', 'netal_fp8_lab.metallib', 'build-receipt.json'):
        if not (root / name).is_file():
            raise ValueError('missing optional lab input: ' + name)
    receipt = json.loads((root / 'build-receipt.json').read_text())
    if not isinstance(receipt, dict) or receipt.get('schema') != 'netal.fp8.matmul-lab.build.v1' or \
            receipt.get('sdk') != 'iphoneos' or version_tuple(receipt.get('deployment_target')) < (27, 0, 0) or \
            receipt.get('models_included') is not False:
        raise ValueError('unsupported optional lab build receipt')
    if version_tuple(receipt['deployment_target']) > version_tuple(deployment_target):
        raise ValueError('the optional lab archive requires a newer deployment target')
    actual = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in ('libnetal_ios_fp8_lab.a', 'NetalFP8MatmulLab.h', 'netal_fp8_lab.metallib')}
    sources = receipt.get('sources')
    if not isinstance(sources, list) or not sources or any(
            not isinstance(row, dict) or not isinstance(row.get('path'), str) or
            not row['path'].startswith('src/ios_fp8_lab/') or '..' in Path(row['path']).parts or
            not isinstance(row.get('sha256'), str) or
            not re.fullmatch('[0-9a-f]{64}', row.get('sha256', '')) for row in sources):
        raise ValueError('invalid optional lab source identity')
    headers = [row['sha256'] for row in sources if row['path'] == 'src/ios_fp8_lab/NetalFP8MatmulLab.h']
    if receipt.get('archive') != 'libnetal_ios_fp8_lab.a' or receipt.get('library') != 'netal_fp8_lab.metallib' or \
            receipt.get('archive_sha256') != actual['libnetal_ios_fp8_lab.a'] or \
            receipt.get('library_sha256') != actual['netal_fp8_lab.metallib'] or headers != [actual['NetalFP8MatmulLab.h']]:
        raise ValueError('optional lab artifact identity mismatch')
    identity = {'schema': receipt['schema'], 'archive_sha256': actual['libnetal_ios_fp8_lab.a'],
                'metallib_sha256': actual['netal_fp8_lab.metallib'], 'header_sha256': actual['NetalFP8MatmulLab.h'],
                'reported_sources': sources, 'models_included': False,
                'build_receipt_sha256': hashlib.sha256((root / 'build-receipt.json').read_bytes()).hexdigest()}
    return root, identity


def load_metalfx_lab(root, deployment_target):
    root = Path(root).absolute()
    for name in ('libnetal_ios_metalfx_lab.a', 'NetalMetalFXLab.h', 'build-receipt.json'):
        if not (root / name).is_file():
            raise ValueError('missing optional MetalFX lab input: ' + name)
    receipt = json.loads((root / 'build-receipt.json').read_text())
    if not isinstance(receipt, dict) or receipt.get('schema') != 'netal.metalfx.lab.build.v1' or \
            receipt.get('sdk') != 'iphoneos' or receipt.get('models_included') is not False or \
            version_tuple(receipt.get('deployment_target')) < (26, 0, 0) or \
            version_tuple(receipt['deployment_target']) > version_tuple(deployment_target):
        raise ValueError('unsupported optional MetalFX lab build receipt')
    sources = receipt.get('sources')
    shared_path = 'src/ios_probe/NetalProbeMetrics.h'
    if not isinstance(sources, list) or not sources or any(
            not isinstance(row, dict) or not isinstance(row.get('path'), str) or
            not (row['path'].startswith('src/ios_metalfx_lab/') or row['path'] == shared_path) or
            '..' in Path(row['path']).parts or not isinstance(row.get('sha256'), str) or
            not re.fullmatch('[0-9a-f]{64}', row['sha256']) for row in sources):
        raise ValueError('invalid optional MetalFX source identity')
    archive_hash = hashlib.sha256((root / 'libnetal_ios_metalfx_lab.a').read_bytes()).hexdigest()
    header_hash = hashlib.sha256((root / 'NetalMetalFXLab.h').read_bytes()).hexdigest()
    shared_hash = hashlib.sha256((ROOT / shared_path).read_bytes()).hexdigest()
    if receipt.get('archive') != 'libnetal_ios_metalfx_lab.a' or receipt.get('archive_sha256') != archive_hash or \
            [row['sha256'] for row in sources if row['path'] == 'src/ios_metalfx_lab/NetalMetalFXLab.h'] != [header_hash] or \
            [row['sha256'] for row in sources if row['path'] == shared_path] != [shared_hash]:
        raise ValueError('MetalFX archive/header/shared-stimulus identity mismatch')
    return root, {'schema': receipt['schema'], 'archive_sha256': archive_hash, 'header_sha256': header_hash,
                  'reported_sources': sources, 'models_included': False,
                  'build_receipt_sha256': hashlib.sha256((root / 'build-receipt.json').read_bytes()).hexdigest()}


def info_plist(bundle_id, deployment_target, fp8_identity=None, metalfx_identity=None, enable_metal_capture=False):
    if fp8_identity and version_tuple(deployment_target) < (27, 0, 0):
        raise ValueError('the optional operator lab requires iOS 27 or newer')
    result = {
        'CFBundleIdentifier': bundle_id, 'CFBundleName': 'NetalSRProbe',
        'CFBundleDisplayName': 'Netal SR Test', 'CFBundleExecutable': 'NetalSRProbe',
        'CFBundlePackageType': 'APPL', 'CFBundleShortVersionString': '0.1',
        'CFBundleVersion': '1', 'MinimumOSVersion': deployment_target,
        'LSRequiresIPhoneOS': True, 'UIDeviceFamily': [1, 2],
        'UIFileSharingEnabled': True, 'LSSupportsOpeningDocumentsInPlace': True,
        'UILaunchScreen': {},
        'MetalCaptureEnabled': bool(enable_metal_capture),
        'NetalProbeSourceSHA256': {
            name: hashlib.sha256((ROOT / 'src/ios_probe' / name).read_bytes()).hexdigest()
            for name in SOURCES},
        'UIApplicationSceneManifest': {
            'UIApplicationSupportsMultipleScenes': False,
            'UISceneConfigurations': {
                'UIWindowSceneSessionRoleApplication': [{
                    'UISceneConfigurationName': 'Default Configuration',
                    'UISceneDelegateClassName': 'NetalProbeSceneDelegate',
                }],
            },
        },
        'UISupportedInterfaceOrientations': [
            'UIInterfaceOrientationPortrait', 'UIInterfaceOrientationLandscapeLeft',
            'UIInterfaceOrientationLandscapeRight'],
    }
    if fp8_identity:
        result['NetalFP8LabIdentity'] = fp8_identity
    if metalfx_identity:
        result['NetalMetalFXLabIdentity'] = metalfx_identity
    if enable_metal_capture and not metalfx_identity:
        raise ValueError('capture builds require the optional MetalFX lab')
    return result


def xcode_spec(bundle_id, deployment_target, team_id=None, fp8_lab=False, metalfx_lab=False):
    if fp8_lab and version_tuple(deployment_target) < (27, 0, 0):
        raise ValueError('the optional operator lab requires iOS 27 or newer')
    settings = {
        'PRODUCT_BUNDLE_IDENTIFIER': bundle_id, 'PRODUCT_NAME': 'NetalSRProbe',
        'SDKROOT': 'iphoneos', 'SUPPORTED_PLATFORMS': 'iphoneos', 'ARCHS': 'arm64',
        'IPHONEOS_DEPLOYMENT_TARGET': deployment_target, 'CLANG_CXX_LANGUAGE_STANDARD': 'c++20',
        'CLANG_ENABLE_OBJC_ARC': 'YES', 'INFOPLIST_FILE': 'Info.plist',
        'GENERATE_INFOPLIST_FILE': 'NO', 'HEADER_SEARCH_PATHS': '$(SRCROOT)/Library/include',
        'LIBRARY_SEARCH_PATHS': '$(SRCROOT)/Library/lib',
        'OTHER_LDFLAGS': ['$(inherited)', '-ObjC'],
        'GCC_PREPROCESSOR_DEFINITIONS': ['$(inherited)', 'NDEBUG=1'],
        'CODE_SIGNING_ALLOWED': 'YES' if team_id else 'NO',
    }
    if team_id:
        settings.update({'DEVELOPMENT_TEAM': team_id, 'CODE_SIGN_STYLE': 'Automatic'})
    dependencies = [{'framework': 'Library/lib/libNetalIOSSR.a', 'embed': False},
                    {'sdk': 'UIKit.framework'}, {'sdk': 'Foundation.framework'}, {'sdk': 'Metal.framework'}]
    sources = [{'path': 'Sources'}]
    if fp8_lab:
        settings['HEADER_SEARCH_PATHS'] = ['$(SRCROOT)/Library/include', '$(SRCROOT)/FP8Lab/include']
        settings['LIBRARY_SEARCH_PATHS'] = ['$(SRCROOT)/Library/lib', '$(SRCROOT)/FP8Lab/lib']
        settings['GCC_PREPROCESSOR_DEFINITIONS'].append('NETAL_IOS_FP8_LAB=1')
        dependencies.append({'framework': 'FP8Lab/lib/libnetal_ios_fp8_lab.a', 'embed': False})
        sources.append({'path': 'Resources', 'buildPhase': 'resources'})
    if metalfx_lab:
        headers = settings['HEADER_SEARCH_PATHS']
        libraries = settings['LIBRARY_SEARCH_PATHS']
        settings['HEADER_SEARCH_PATHS'] = (headers if isinstance(headers, list) else [headers]) + ['$(SRCROOT)/MetalFXLab/include']
        settings['LIBRARY_SEARCH_PATHS'] = (libraries if isinstance(libraries, list) else [libraries]) + ['$(SRCROOT)/MetalFXLab/lib']
        settings['GCC_PREPROCESSOR_DEFINITIONS'].append('NETAL_IOS_METALFX_LAB=1')
        dependencies += [{'framework': 'MetalFXLab/lib/libnetal_ios_metalfx_lab.a', 'embed': False}, {'sdk': 'MetalFX.framework'}]
    return {'name': 'NetalSRProbe', 'options': {'deploymentTarget': {'iOS': deployment_target}},
            'targets': {'NetalSRProbe': {
                'type': 'application', 'platform': 'iOS', 'sources': sources,
                'settings': {'base': settings},
                'dependencies': dependencies}}}


def build(library_root, output, sdk='iphoneos', deployment_target='26.0',
          bundle_id='org.netal.srprobe', prepare_xcode=False, team_id=None,
          sign_identity=None, provisioning_profile=None, fp8_lab_root=None, metalfx_lab_root=None, enable_metal_capture=False):
    library_root, output = validate(library_root, output, sdk, deployment_target, bundle_id, team_id)
    fp8_identity = None
    if fp8_lab_root:
        fp8_lab_root, fp8_identity = load_fp8_lab(fp8_lab_root, deployment_target)
    metalfx_identity = None
    if metalfx_lab_root:
        metalfx_lab_root, metalfx_identity = load_metalfx_lab(metalfx_lab_root, deployment_target)
    if enable_metal_capture and not metalfx_lab_root:
        raise ValueError('--enable-metal-capture requires --metalfx-lab-root')
    if team_id and not prepare_xcode:
        raise ValueError('--team-id requires --prepare-xcode for caller-managed provisioning')
    if provisioning_profile and not sign_identity:
        raise ValueError('--provisioning-profile requires --sign-identity')
    if provisioning_profile and not Path(provisioning_profile).is_file():
        raise ValueError('provisioning profile is not a file')
    if prepare_xcode and not shutil.which('xcodegen'):
        raise ValueError('--prepare-xcode requires xcodegen')
    sdk_root = subprocess.check_output(['xcrun', '--sdk', sdk, '--show-sdk-path'], text=True).strip()
    sdk_version = subprocess.check_output(['xcrun', '--sdk', sdk, '--show-sdk-version'], text=True).strip()
    if version_tuple(deployment_target) > version_tuple(sdk_version):
        raise ValueError('deployment target is newer than the installed SDK')
    if fp8_lab_root and version_tuple(sdk_version) < (27, 0, 0):
        raise ValueError('the optional operator lab requires the physical iPhone SDK 27 or newer')
    triple = 'arm64-apple-ios' + deployment_target
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.netal-ios-probe-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'output'
        stage.mkdir()
        app = stage / 'NetalSRProbe.app'
        app.mkdir()
        obj = Path(temporary) / 'NetalSRProbe.o'
        flags = ['-target', triple, '-isysroot', sdk_root, '-fobjc-arc', '-std=c++20',
                 '-O2', '-Wall', '-Wextra', '-Werror', '-Wno-deprecated-declarations',
                 '-I', str(library_root / 'include')]
        if fp8_lab_root:
            flags += ['-DNETAL_IOS_FP8_LAB=1', '-I', str(fp8_lab_root)]
        if metalfx_lab_root:
            flags += ['-DNETAL_IOS_METALFX_LAB=1', '-I', str(metalfx_lab_root)]
        subprocess.run(['xcrun', '--sdk', sdk, 'clang++', *flags, '-c',
                        str(ROOT / 'src/ios_probe/NetalSRProbe.mm'), '-o', str(obj)], check=True)
        check_macho_platform(obj, 2, deployment_target)
        binary = app / 'NetalSRProbe'
        extra_libraries = [str(fp8_lab_root / 'libnetal_ios_fp8_lab.a')] if fp8_lab_root else []
        extra_frameworks = []
        if metalfx_lab_root:
            extra_libraries.append(str(metalfx_lab_root / 'libnetal_ios_metalfx_lab.a'))
            extra_frameworks = ['-framework', 'MetalFX']
        subprocess.run(['xcrun', '--sdk', sdk, 'clang++', '-target', triple, '-isysroot', sdk_root,
                        str(obj), str(library_root / 'lib/libNetalIOSSR.a'), '-ObjC',
                        *extra_libraries,
                        '-framework', 'UIKit', '-framework', 'Foundation', '-framework', 'Metal',
                        *extra_frameworks,
                        '-o', str(binary)], check=True)
        with (app / 'Info.plist').open('wb') as stream:
            plistlib.dump(info_plist(bundle_id, deployment_target, fp8_identity, metalfx_identity, enable_metal_capture), stream)
        if fp8_lab_root:
            shutil.copy2(fp8_lab_root / 'netal_fp8_lab.metallib', app / 'netal_fp8_lab.metallib')
        if provisioning_profile:
            shutil.copyfile(provisioning_profile, app / 'embedded.mobileprovision')
        if sign_identity:
            subprocess.run(['codesign', '--force', '--sign', sign_identity, str(app)], check=True)
            subprocess.run(['codesign', '--verify', '--strict', str(app)], check=True)
        if prepare_xcode:
            project_root = stage / 'Xcode'
            (project_root / 'Sources').mkdir(parents=True)
            for filename in SOURCES:
                shutil.copy2(ROOT / 'src/ios_probe' / filename, project_root / 'Sources' / filename)
            shutil.copytree(library_root / 'include', project_root / 'Library/include')
            (project_root / 'Library/lib').mkdir(parents=True)
            shutil.copy2(library_root / 'lib/libNetalIOSSR.a', project_root / 'Library/lib/libNetalIOSSR.a')
            if fp8_lab_root:
                (project_root / 'FP8Lab/include').mkdir(parents=True)
                (project_root / 'FP8Lab/lib').mkdir(parents=True)
                (project_root / 'Resources').mkdir()
                shutil.copy2(fp8_lab_root / 'NetalFP8MatmulLab.h', project_root / 'FP8Lab/include/NetalFP8MatmulLab.h')
                shutil.copy2(fp8_lab_root / 'libnetal_ios_fp8_lab.a', project_root / 'FP8Lab/lib/libnetal_ios_fp8_lab.a')
                shutil.copy2(fp8_lab_root / 'netal_fp8_lab.metallib', project_root / 'Resources/netal_fp8_lab.metallib')
            if metalfx_lab_root:
                (project_root / 'MetalFXLab/include').mkdir(parents=True)
                (project_root / 'MetalFXLab/lib').mkdir(parents=True)
                shutil.copy2(metalfx_lab_root / 'NetalMetalFXLab.h', project_root / 'MetalFXLab/include/NetalMetalFXLab.h')
                shutil.copy2(metalfx_lab_root / 'libnetal_ios_metalfx_lab.a', project_root / 'MetalFXLab/lib/libnetal_ios_metalfx_lab.a')
            with (project_root / 'Info.plist').open('wb') as stream:
                plistlib.dump(info_plist(bundle_id, deployment_target, fp8_identity, metalfx_identity, enable_metal_capture), stream)
            spec = project_root / 'project.json'
            spec.write_text(json.dumps(xcode_spec(bundle_id, deployment_target, team_id, bool(fp8_lab_root), bool(metalfx_lab_root)), indent=2) + '\n')
            subprocess.run(['xcodegen', 'generate', '--spec', str(spec)], cwd=project_root, check=True)
        source_hashes = {name: hashlib.sha256((ROOT / 'src/ios_probe' / name).read_bytes()).hexdigest()
                         for name in SOURCES}
        receipt = {
            'schema': 'netal.ios.sr.probe-build.v1', 'sdk': sdk, 'sdk_version': sdk_version,
            'cpu_target': triple, 'deployment_target': deployment_target, 'object_platform': 2,
            'bundle_identifier': bundle_id, 'app': app.name,
            'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
            'library_sha256': hashlib.sha256((library_root / 'lib/libNetalIOSSR.a').read_bytes()).hexdigest(),
            'source_sha256': source_hashes, 'signed': bool(sign_identity),
            'optional_fp8_lab': fp8_identity,
            'optional_metalfx_lab': metalfx_identity, 'metal_capture_enabled': bool(enable_metal_capture),
            'xcode_project_prepared': prepare_xcode, 'embedded_model_count': 0,
            'installed': False, 'device_execution_validated': False, 'latency_validated': False,
            'runtime_location': 'Documents/runtime', 'config_location': 'Documents/benchmark.json',
            'device_receipt_location': 'Documents/probe-receipt.json',
        }
        (stage / 'build-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        (stage / 'benchmark.example.json').write_text(json.dumps({
            'profile': '1280x720', 'backend': 'batched', 'metal_api': 'metal4',
            'warmup_frames': 3, 'measured_frames': 60, 'save_last_output': False,
            'fast_input_boundary': False, 'diagnostic_output_repeat': 1}, indent=2) + '\n')
        if fp8_lab_root:
            (stage / 'benchmark.fp8-lab.example.json').write_text(json.dumps({
                'probe_mode': 'fp8_lab', 'fp8_sample_count': 3, 'fp8_repetitions_per_sample': 8}, indent=2) + '\n')
        if metalfx_lab_root:
            for mode in ('spatial', 'temporal'):
                example = {'probe_mode': 'metalfx_' + mode, 'render_size': '640x360', 'profile': '1280x720',
                           'warmup_frames': 8, 'measured_frames': 60, 'capture': bool(enable_metal_capture)}
                (stage / ('benchmark.metalfx-' + mode + '.example.json')).write_text(json.dumps(example, indent=2) + '\n')
        stage.rename(output)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sdk', choices=('iphoneos',), default='iphoneos')
    parser.add_argument('--deployment-target', default='26.0')
    parser.add_argument('--bundle-id', default='org.netal.srprobe')
    parser.add_argument('--prepare-xcode', action='store_true')
    parser.add_argument('--team-id')
    parser.add_argument('--sign-identity')
    parser.add_argument('--provisioning-profile', type=Path)
    parser.add_argument('--fp8-lab-root', type=Path, help='optional separately built, model-free iOS 27 operator lab')
    parser.add_argument('--metalfx-lab-root', type=Path, help='optional model-free Metal 4/MetalFX comparison lab')
    parser.add_argument('--enable-metal-capture', action='store_true', help='dedicated capture build; ordinary performance measurements disabled')
    args = parser.parse_args()
    try:
        receipt = build(args.library_root, args.output, args.sdk, args.deployment_target,
                        args.bundle_id, args.prepare_xcode, args.team_id,
                        args.sign_identity, args.provisioning_profile, args.fp8_lab_root, args.metalfx_lab_root, args.enable_metal_capture)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print('result=fail ' + str(error), file=sys.stderr)
        return 1
    print('result=pass app=' + receipt['app'] + ' device_execution_validated=false')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
