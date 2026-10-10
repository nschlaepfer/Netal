#!/usr/bin/env python3
"""Assemble a rebuilt FG engine with explicitly supplied, identity-checked inputs.

This does not extract models, execute inference, or establish redistribution rights.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'src/fg_runtime/runtime-inputs.json'


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def load_json(path):
    return json.loads(path.read_text(), object_pairs_hook=unique_object)


def identity(data):
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def checked_file(path, expected):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Missing file or symlink: {path.name}')
    data = path.read_bytes()
    if (not isinstance(expected, dict) or set(expected) != {'bytes', 'sha256'}
            or type(expected['bytes']) is not int or expected['bytes'] <= 0
            or not isinstance(expected['sha256'], str)
            or not re.fullmatch('[0-9a-f]{64}', expected['sha256'])
            or identity(data) != expected):
        raise ValueError(f'Identity mismatch: {path.name}')
    return data


def file_set(root):
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Input must be a real directory')
    result = set()
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError(f'Symlink in input: {path.name}')
        if path.is_file():
            result.add(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise ValueError(f'Non-regular input: {path.name}')
    return result


def safe_relative(name):
    path = PurePosixPath(name)
    return (bool(re.fullmatch(r'[A-Za-z0-9_.\-/]+', name)) and not path.is_absolute()
            and path.as_posix() == name and all(p not in ('', '.', '..') for p in name.split('/')))


def validate_engine(engine):
    actual = file_set(engine)
    receipt = load_json(engine / 'build-receipt.json')
    if (receipt.get('schema') != 'netal.fg.source-engine.v1'
            or receipt.get('profile') != '640x360' or receipt.get('hmma') != 'u32'
            or receipt.get('ex2') != 'compact' or receipt.get('live_zero') is not True):
        raise ValueError('Unsupported engine recipe')
    files = receipt.get('files')
    if not isinstance(files, dict) or not all(isinstance(p, str) and safe_relative(p) for p in files):
        raise ValueError('Invalid engine inventory')
    if actual != set(files) | {'build-receipt.json'}:
        raise ValueError('Engine inventory is incomplete or has extra files')
    recipe = load_json(ROOT / 'src/fg_runtime/kernel-build.json')
    required = {x['name'] + '.metallib' for x in recipe['kernels']}
    # These variants are selected by the game path's hardware-prefix mode.
    required.update(name + '_hw.metallib' for name in (
        'conv_3x3', 'conv_1x1', 'conv_pre', 'conv0', 'conv1', 'conv2',
        'bot0', 'bot1', 'block1_conv0', 'block1_conv1', 'block1_conv2',
        'block1_bot0', 'block1_bot1',
    ))
    kernels = receipt.get('kernels')
    if not isinstance(kernels, list) or not all(isinstance(k, dict) for k in kernels):
        raise ValueError('Invalid engine kernel list')
    names = [k.get('library') for k in kernels]
    libraries = {p.removeprefix('libs/') for p in files if p.startswith('libs/')}
    if (any(not isinstance(name, str) for name in names) or len(names) != len(set(names))
            or set(names) != libraries or required != libraries):
        raise ValueError('Engine libraries do not cover the build recipe')
    required_host = {'include/fg_runtime.h', 'lib/libnetal_fg_runtime.a',
                     'lib/libnetal_fg_runtime-arm64.a', 'lib/libnetal_fg_runtime-x86_64.a'}
    if not required_host <= set(files):
        raise ValueError('Engine host archives/header missing')
    payload = {}
    for name, expected in files.items():
        data = checked_file(engine / name, expected)
        if name.startswith('libs/') or name in required_host:
            payload[name] = data
    for kernel in kernels:
        if kernel.get('sha256') != files['libs/' + kernel['library']]['sha256']:
            raise ValueError('Kernel and file inventories disagree')
    return payload


def assemble(engine, model, math_paths, output):
    if output.exists() or output.is_symlink():
        raise ValueError('Output exists; use a new directory')
    for source in (engine, model):
        if output.resolve().is_relative_to(source.resolve()):
            raise ValueError('Output must be outside the engine and model inputs')
    contract = load_json(CONTRACT)
    payload = validate_engine(engine)
    if file_set(model) != set(contract['model']):
        raise ValueError('Model inventory must contain exactly the 67 supported buffers')
    for name, expected in contract['model'].items():
        payload['model/' + name] = checked_file(model / name, expected)
    for name, expected in contract['math'].items():
        payload[name] = checked_file(math_paths[name], expected)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.netal-fg-assembly-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'bundle'
        stage.mkdir()
        for name, data in payload.items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        manifest = {
            'schema': 'netal.fg.runtime-bundle.v1', 'profile': '640x360',
            'assembly': 'source-engine-explicit-inputs-v1',
            'source_engine_receipt_sha256': identity((engine / 'build-receipt.json').read_bytes())['sha256'],
            'input_contract_sha256': identity(CONTRACT.read_bytes())['sha256'],
            'backend_receipt': None, 'execution_validated': False,
            'gameplay_validated': False, 'redistribution_clearance': False,
            'files': {name: identity(data) for name, data in sorted(payload.items())},
        }
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        if output.exists() or output.is_symlink():
            raise ValueError('Output appeared during assembly; preserved')
        stage.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('engine', 'model', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('sigmoid', 'ex2-unit', 'ex2-fraction'):
        parser.add_argument('--' + name, type=Path,
                            default=ROOT / 'runtime/fg-arithmetic-calibration' / (name + '.bin'))
    parser.add_argument('--rcp', type=Path,
                        default=ROOT / 'runtime/arithmetic-calibration/rcp-correction-rom.bin')
    args = parser.parse_args()
    try:
        manifest = assemble(args.engine, args.model, {
            'rom.bin': args.rcp, 'sigmoid.bin': args.sigmoid,
            'ex2-unit.bin': args.ex2_unit, 'ex2-fraction.bin': args.ex2_fraction,
        }, args.output.absolute())
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f'assembly failed: {error}\n')
    print(f"result=pass files={len(manifest['files'])} execution_validated=false")


if __name__ == '__main__':
    main()
