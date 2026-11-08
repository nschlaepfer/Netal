"""Shared, explicit Apple SDK/target selection for source-only SR builds."""
import re
import struct
import subprocess
from pathlib import Path

SDKS = ('macosx', 'iphoneos', 'iphonesimulator')


def version_tuple(value):
    if not re.fullmatch(r'[0-9]+\.[0-9]+(?:\.[0-9]+)?', value or ''):
        raise ValueError('deployment target must be major.minor or major.minor.patch')
    parts = tuple(int(p) for p in value.split('.'))
    if any(p > 255 for p in parts):
        raise ValueError('deployment target components must be <=255')
    return parts + (0,) * (3 - len(parts))


def add_metal_target_arguments(parser):
    parser.add_argument('--sdk', choices=SDKS, default='macosx',
                        help='Apple SDK (default preserves the existing macOS build)')
    parser.add_argument('--deployment-target',
                        help='minimum OS; iOS defaults to 26.0, macOS keeps the compiler default')


def metal_target(sdk, deployment_target=None):
    if sdk not in SDKS:
        raise ValueError('unsupported Apple SDK')
    if sdk != 'macosx':
        deployment_target = deployment_target or '26.0'
        if version_tuple(deployment_target) < (26, 0, 0):
            raise ValueError('iOS SR builds require iOS 26.0 or newer')
        suffix = '-simulator' if sdk == 'iphonesimulator' else ''
        return deployment_target, f'air64-apple-ios{deployment_target}{suffix}'
    if deployment_target:
        version_tuple(deployment_target)
        return deployment_target, f'air64-apple-macosx{deployment_target}'
    return None, None


def metal_toolchain(sdk='macosx', deployment_target=None):
    deployment_target, triple = metal_target(sdk, deployment_target)
    # Keep the old no-argument macOS command and target behavior unchanged.
    prefix = ['xcrun'] if triple is None else ['xcrun', '--sdk', sdk]
    compiler = Path(subprocess.check_output(prefix + ['--find', 'metal'], text=True).strip())
    linker = compiler.with_name('metallib')
    if not linker.is_file():
        linker = Path(subprocess.check_output(prefix + ['--find', 'metallib'], text=True).strip())
    flags = []
    receipt = {'sdk': sdk, 'deployment_target': deployment_target, 'metal_target': triple}
    if triple:
        sdk_root = subprocess.check_output(['xcrun', '--sdk', sdk, '--show-sdk-path'], text=True).strip()
        sdk_version = subprocess.check_output(['xcrun', '--sdk', sdk, '--show-sdk-version'], text=True).strip()
        if version_tuple(deployment_target) > version_tuple(sdk_version):
            raise ValueError('deployment target is newer than the installed SDK')
        flags = ['-target', triple, '-isysroot', sdk_root]
        receipt['sdk_version'] = sdk_version
    return compiler, linker, flags, receipt


def check_macho_platform(path, expected_platform, deployment_target):
    """Reject a macOS/simulator object accidentally mixed into an iPhone archive."""
    data = Path(path).read_bytes()
    if len(data) < 32 or struct.unpack_from('<I', data)[0] != 0xfeedfacf:
        raise ValueError('compiler output is not a 64-bit Mach-O object')
    cpu, subtype, kind = struct.unpack_from('<III', data, 4)
    if cpu != 0x100000c or kind != 1:  # CPU_TYPE_ARM64, MH_OBJECT
        raise ValueError('compiler output is not an arm64 Mach-O object')
    count, size = struct.unpack_from('<II', data, 16)
    end = 32 + size
    if end > len(data) or count > size // 8:
        raise ValueError('invalid Mach-O load commands')
    offset = 32
    want = version_tuple(deployment_target)
    wanted_min = (want[0] << 16) | (want[1] << 8) | want[2]
    platforms = []
    for _ in range(count):
        if offset + 8 > end:
            raise ValueError('truncated Mach-O load command')
        command, length = struct.unpack_from('<II', data, offset)
        if length < 8 or offset + length > end:
            raise ValueError('invalid Mach-O load-command length')
        if command == 0x32:  # LC_BUILD_VERSION
            if length < 24:
                raise ValueError('truncated LC_BUILD_VERSION')
            platform, minimum = struct.unpack_from('<II', data, offset + 8)
            platforms.append(platform)
            if platform != expected_platform or minimum != wanted_min:
                raise ValueError('compiler output has the wrong platform or deployment target')
        offset += length
    if offset != end or platforms != [expected_platform]:
        raise ValueError('compiler output lacks one unambiguous LC_BUILD_VERSION')
