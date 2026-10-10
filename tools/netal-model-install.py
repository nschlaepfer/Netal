#!/usr/bin/env python3
"""Install the supported SR model atomically into an existing runtime skeleton.

This verifies full model hashes, not rendered frames. It does not bootstrap engine assets or
certify runtime/gameplay correctness. Unknown model identities are not installed.
"""
import argparse
import contextlib
import ctypes
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
def runtime_manifest_module():
    spec = importlib.util.spec_from_file_location('sr_runtime_manifest', ROOT / 'scripts/netal_sr_runtime_manifest.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
_spec = importlib.util.spec_from_file_location('netal_extract', Path(__file__).with_name('netal-dlss-extract.py'))
extract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extract)


def atomic_exchange(first, second):
    """Swap existing sibling directories as one filesystem operation; no two-rename fallback."""
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == 'darwin':
        function = library.renamex_np
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(os.fsencode(first), os.fsencode(second), 2)  # RENAME_SWAP
    elif sys.platform.startswith('linux') and hasattr(library, 'renameat2'):
        function = library.renameat2
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(-100, os.fsencode(first), -100, os.fsencode(second), 2)  # RENAME_EXCHANGE
    else:
        raise ValueError('This platform does not provide the required atomic directory exchange')
    if result:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


@contextlib.contextmanager
def install_lock(bundle):
    # The lock is outside the directory being exchanged, so all callers lock the same inode.
    lock_path = bundle.parent / ('.' + bundle.name + '.install.lock')
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another model installation is in progress') from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    # Keep the empty lock file: unlinking it could let another caller lock a different inode.


def validate_model(model):
    files = {}
    for name, expected in extract.REFERENCE['files'].items():
        path = model / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('Missing or linked model file: ' + name)
        if path.stat().st_size != extract.KNOWN[name] or extract.sha(path) != expected:
            raise ValueError('Model does not match the supported reference: ' + name)
        files['model/' + name] = {'sha256': expected, 'bytes': extract.KNOWN[name]}
    return files


def validate_skeleton(bundle):
    if bundle.is_symlink() or not bundle.is_dir():
        raise ValueError('Select a real runtime bundle directory')
    if any(path.is_symlink() for path in bundle.rglob('*')):
        raise ValueError('Runtime skeleton contains symlinks; provision portable files before installing')
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if not isinstance(manifest, dict):
        raise ValueError('Runtime manifest must be an object')
    if manifest.get('model') != extract.REFERENCE['model']:
        raise ValueError('Runtime skeleton targets a different or unidentified model')
    entries = manifest.get('files')
    if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
        raise ValueError('Runtime manifest has no file inventory')
    names = [entry.get('path') for entry in entries]
    if any(not isinstance(name, str) for name in names) or len(names) != len(set(names)):
        raise ValueError('Runtime manifest has invalid or duplicate entries')
    for name in names:
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Runtime manifest contains an unsafe path')
    if manifest.get('schema') == 'netal.sr.runtime.v2':
        runtime_manifest_module().verify_engine(bundle, manifest)
    return manifest


def install(model, bundle, receipt, dry_run=False):
    manifest = validate_skeleton(bundle)
    entries = validate_model(model)
    if receipt.get('source_dll_sha256') != extract.REFERENCE['source_dll_sha256'] or not receipt.get('matches_reference_model_bytes'):
        raise ValueError('Extraction receipt does not identify the supported source runtime')
    abi = None
    if (model / 'abi').exists():
        if (model / 'abi').is_symlink(): raise ValueError('Linked boundary metadata is not accepted')
        abi = extract.abi_module().validate_package(model / 'abi')
    if dry_run:
        return
    with tempfile.TemporaryDirectory(prefix='.' + bundle.name + '.install-', dir=bundle.parent) as temporary:
        stage = Path(temporary) / 'bundle'
        shutil.copytree(bundle, stage)
        target = stage / 'model'
        target.mkdir(exist_ok=True)
        for name in extract.KNOWN:
            shutil.copyfile(model / name, target / name)
        if abi is not None:
            abi_target = target / 'abi'
            abi_target.mkdir(exist_ok=True)  # Preserve unrelated user sidecars in the staged copy.
            templates, metadata = abi
            for name, data in templates.items(): (abi_target / name).write_bytes(data)
            (abi_target / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
        # Reverify staged bytes before the only mutation of the installed bundle.
        validate_model(target)
        kept = [entry for entry in manifest['files'] if entry['path'] not in entries]
        manifest['files'] = sorted(kept + [dict(path=name, **value) for name, value in entries.items()], key=lambda row: row['path'])
        manifest['weights_sha256'] = entries['model/weights.bin']['sha256']
        manifest['ingest'] = {
            'source_dll_sha256': receipt['source_dll_sha256'],
            'matches_reference_model_bytes': True,
            'verified_against_rtx_receipts': False,
            'inference_validated_by_install': False,
            'gameplay_validated_by_install': False,
        }
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        (target / 'extraction-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        if manifest.get('schema') == 'netal.sr.runtime.v2':
            runtime_manifest_module().seal(stage, replace=True)
        # A process interruption before this leaves the old tree intact; after it the complete
        # new tree is active. The old tree stays at stage until normal cleanup (or as recovery data
        # if the process is killed). Engine assets are copied, not changed by the model installer.
        atomic_exchange(stage, bundle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dll', type=Path)
    parser.add_argument('--bundle', type=Path, default=ROOT / 'work/re9-sr-runtime-bundle')
    parser.add_argument('--capture-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        bundle = args.bundle.absolute()
        validate_skeleton(bundle)
        capture = args.capture_dir
        if capture is None:
            capture = Path(os.environ.get('NETAL_CAPTURE_DIR', ROOT / 'build/model-capture'))
            if not capture.exists() and 'NETAL_CAPTURE_DIR' not in os.environ:
                capture = ROOT / 'vendor/dlss-capture'
        with install_lock(bundle):
            with tempfile.TemporaryDirectory(prefix='netal-ingest-') as temporary:
                model = Path(temporary) / 'model'
                subprocess.run([sys.executable, str(Path(__file__).with_name('netal-dlss-extract.py')),
                                str(args.dll.resolve(strict=True)), '--capture-dir', str(capture.resolve()), '--out', str(model)], check=True)
                print('verifying: full supported-model hashes', flush=True)
                receipt = json.loads((model / 'extraction-receipt.json').read_text())
                install(model, bundle, receipt, args.dry_run)
        print('result=pass ' + ('dry-run; installed bundle unchanged' if args.dry_run else 'supported model bytes installed'))
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print('result=fail ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
