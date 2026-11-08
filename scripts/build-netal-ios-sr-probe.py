#!/usr/bin/env python3
"""Build an unsigned physical-iPhone UIKit SR probe; caller supplies runtime and signing.

The library root is the output of build-netal-ios-sr.py. No model, game data,
device identifier, signing identity or provisioning profile is built in.
--prepare-xcode also makes a self-contained project for caller-managed signing.
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
SOURCES = ('NetalSRProbe.mm', 'NetalProbeMetrics.h')


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


def info_plist(bundle_id, deployment_target):
    return {
        'CFBundleIdentifier': bundle_id, 'CFBundleName': 'NetalSRProbe',
        'CFBundleDisplayName': 'Netal SR Test', 'CFBundleExecutable': 'NetalSRProbe',
        'CFBundlePackageType': 'APPL', 'CFBundleShortVersionString': '0.1',
        'CFBundleVersion': '1', 'MinimumOSVersion': deployment_target,
        'LSRequiresIPhoneOS': True, 'UIDeviceFamily': [1, 2],
        'UIFileSharingEnabled': True, 'LSSupportsOpeningDocumentsInPlace': True,
        'UILaunchScreen': {},
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


def xcode_spec(bundle_id, deployment_target, team_id=None):
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
    return {'name': 'NetalSRProbe', 'options': {'deploymentTarget': {'iOS': deployment_target}},
            'targets': {'NetalSRProbe': {
                'type': 'application', 'platform': 'iOS', 'sources': [{'path': 'Sources'}],
                'settings': {'base': settings},
                'dependencies': [{'framework': 'Library/lib/libNetalIOSSR.a', 'embed': False},
                                 {'sdk': 'UIKit.framework'}, {'sdk': 'Foundation.framework'},
                                 {'sdk': 'Metal.framework'}]}}}


def build(library_root, output, sdk='iphoneos', deployment_target='26.0',
          bundle_id='org.netal.srprobe', prepare_xcode=False, team_id=None,
          sign_identity=None, provisioning_profile=None):
    library_root, output = validate(library_root, output, sdk, deployment_target, bundle_id, team_id)
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
        subprocess.run(['xcrun', '--sdk', sdk, 'clang++', *flags, '-c',
                        str(ROOT / 'src/ios_probe/NetalSRProbe.mm'), '-o', str(obj)], check=True)
        check_macho_platform(obj, 2, deployment_target)
        binary = app / 'NetalSRProbe'
        subprocess.run(['xcrun', '--sdk', sdk, 'clang++', '-target', triple, '-isysroot', sdk_root,
                        str(obj), str(library_root / 'lib/libNetalIOSSR.a'), '-ObjC',
                        '-framework', 'UIKit', '-framework', 'Foundation', '-framework', 'Metal',
                        '-o', str(binary)], check=True)
        with (app / 'Info.plist').open('wb') as stream:
            plistlib.dump(info_plist(bundle_id, deployment_target), stream)
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
            with (project_root / 'Info.plist').open('wb') as stream:
                plistlib.dump(info_plist(bundle_id, deployment_target), stream)
            spec = project_root / 'project.json'
            spec.write_text(json.dumps(xcode_spec(bundle_id, deployment_target, team_id), indent=2) + '\n')
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
            'xcode_project_prepared': prepare_xcode, 'embedded_model_count': 0,
            'installed': False, 'device_execution_validated': False, 'latency_validated': False,
            'runtime_location': 'Documents/runtime', 'config_location': 'Documents/benchmark.json',
            'device_receipt_location': 'Documents/probe-receipt.json',
        }
        (stage / 'build-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        (stage / 'benchmark.example.json').write_text(json.dumps({
            'profile': '1280x720', 'backend': 'batched', 'metal_api': 'metal4',
            'warmup_frames': 3, 'measured_frames': 60, 'save_last_output': False}, indent=2) + '\n')
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
    args = parser.parse_args()
    try:
        receipt = build(args.library_root, args.output, args.sdk, args.deployment_target,
                        args.bundle_id, args.prepare_xcode, args.team_id,
                        args.sign_identity, args.provisioning_profile)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print('result=fail ' + str(error), file=sys.stderr)
        return 1
    print('result=pass app=' + receipt['app'] + ' device_execution_validated=false')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
