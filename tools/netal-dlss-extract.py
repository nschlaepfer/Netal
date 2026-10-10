#!/usr/bin/env python3
"""Extract SR model data locally from a user-supplied DLSS runtime under Wine.

The source-built CUDA recorder captures the DLL's host-to-device uploads; no NVIDIA GPU is
used. Default acceptance is restricted to the hash-pinned quality model below. Matching
model bytes does not establish engine correctness or gameplay performance. No model bytes,
NVIDIA runtime, or SDK headers are included in the public source.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

KNOWN = {'weights.bin': 3545824, 'constants.bin': 524288}
REFERENCE = {
    'source_dll_sha256': 'c85f971ce023c9f3492fc7455f0b01a24ba18ea39636407a846902c4360b0b7e',
    'scenario': 'quality',
    'model': 'DLSS_SR_Feature1_Quality_K',
    'files': {
        'weights.bin': 'ddaa59cd82deed53bad9d65b82b454f1439fc4875e4220b3ffafafb6cd0296a1',
        'constants.bin': '305f4c6773d1cb333b577094e28a46beb1867898716d8290121862a2cb06e773',
    },
}
CAPTURE_FILES = ['dlss-sr-cuda-capture.exe', 'nvcuda.dll', 'nvapi64.dll', 'nvngx.dll', 'nvcuda-forward-trace.dll']
MAX_ARENA_BYTES = 512 * 1024 * 1024


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(stream.read()).hexdigest()


def find_wine():
    override = os.environ.get('NETAL_WINE')
    if override:
        binary = shutil.which(override)
        if not binary:
            raise ValueError('NETAL_WINE does not name an executable')
        return binary
    for candidate in ['wine64', '/usr/local/bin/wine64', '/opt/homebrew/bin/wine64']:
        binary = shutil.which(candidate)
        if binary:
            return binary
    raise ValueError('No standalone wine64 found. Install Wine or set NETAL_WINE. CrossOver requires a bottle and is not used by this extractor.')


def stage_capture(source, work):
    missing = [name for name in CAPTURE_FILES if not (source / name).is_file()]
    if missing:
        raise ValueError('Capture helpers missing: ' + ', '.join(missing) + '. Build with scripts/build-netal-capture-tools.py and pass --capture-dir.')
    staged = work / 'capture'
    staged.mkdir()
    for name in CAPTURE_FILES:
        shutil.copy2(source / name, staged / name)
    return staged


def run_loader(dll, capture, work, scenario, timeout):
    build = stage_capture(capture, work)
    dumps = work / 'dumps'
    dumps.mkdir()
    env = dict(os.environ)
    env.update({
        'WINEPREFIX': str(work / 'wineprefix'), 'WINEDEBUG': '-all',
        'WINEDLLOVERRIDES': 'nvngx,nvcuda,nvapi64=n,b',
        'NEURALSTUDIO_CUDA_TRACE': 'extract-trace.jsonl',
        'NEURALSTUDIO_CUDA_DUMP_DIR': str(dumps),
        'NEURALSTUDIO_CUDA_CAPTURE_SNAPSHOTS': '1',
        'NEURALSTUDIO_CUDA_CAPTURE_DUMP_DIR': str(dumps),
        'NEURALSTUDIO_CUDA_SNAPSHOT_DIR': str(dumps),
    })
    windows_path = 'Z:' + str(dll).replace('/', '\\')
    wine = find_wine()
    server = Path(wine).resolve().with_name('wineserver')
    if not os.access(server, os.X_OK):
        raise ValueError('The selected Wine installation has no executable wineserver for isolated cleanup')
    try:
        result = subprocess.run([wine, './dlss-sr-cuda-capture.exe', windows_path, scenario],
                                cwd=build, env=env, capture_output=True, text=True, timeout=timeout)
    finally:
        # This prefix was created solely for this run. Never terminate a user's game bottle or
        # invoke a global process-name kill when cleaning up the loader's background services.
        stopped = subprocess.run([str(server), '-k'], env=env, capture_output=True, timeout=15)
        # Wine returns 1 for -k when this isolated server has already exited. Waiting is
        # authoritative: -w must succeed before the prefix is removed.
        subprocess.run([str(server), '-w'], env=env, capture_output=True, timeout=15, check=True)
        if stopped.returncode not in (0, 1):
            raise ValueError('Isolated Wine server cleanup failed')
    if result.returncode != 0 or 'result=observed' not in result.stdout.splitlines():
        raise ValueError('DLSS feature creation failed under Wine: ' + (result.stdout + result.stderr)[-2000:])
    return build / 'extract-trace.jsonl', dumps


def arenas(trace):
    allocations, transfers = [], []
    for line in trace.read_text().splitlines():
        record = json.loads(line)
        if record.get('call') == 'cuMemAlloc' and 'pointer' in record:
            allocations.append(record)
        elif record.get('call') == 'cuMemcpyHtoDAsync' and 'destination' in record:
            transfers.append(record)
    return allocations, transfers


def transfer_data(record, dumps):
    length = int(record['bytes'])
    transfer_id = int(record['transfer_id'])
    if length < 0 or length > MAX_ARENA_BYTES or transfer_id < 0:
        raise ValueError('Invalid transfer size or identity')
    blob = dumps / ('transfer-%04d.bin' % transfer_id)
    if not blob.is_file() or blob.stat().st_size != length:
        raise ValueError('Missing or truncated transfer %d' % transfer_id)
    return blob.read_bytes()


def rebuild(base, size, transfers, dumps):
    if base < 0 or not 0 < size <= MAX_ARENA_BYTES:
        raise ValueError('Invalid arena bounds')
    arena, covered = bytearray(size), bytearray(size)
    for record in transfers:
        destination = int(record['destination'], 16)
        length = int(record['bytes'])
        if not (base <= destination < base + size):
            continue
        offset = destination - base
        if length < 0 or offset + length > size:
            raise ValueError('Transfer exceeds its arena')
        data = transfer_data(record, dumps)
        arena[offset:offset + length] = data
        covered[offset:offset + length] = b'\x01' * length
    return None if 0 in covered else bytes(arena)


def select_model(allocations, transfers, dumps, expected=None):
    weights = {}
    for base, size in sorted({(int(a['pointer'], 16), int(a['bytes'])) for a in allocations}):
        if size < 1 << 20:
            continue
        if expected and size != KNOWN['weights.bin']:
            continue
        data = rebuild(base, size, transfers, dumps)
        if data is not None:
            digest = hashlib.sha256(data).hexdigest()
            if not expected or digest == expected['weights.bin']:
                weights[digest] = data
    if len(weights) != 1:
        raise ValueError('No unique complete weights arena matches the requested model')
    constants = {}
    for record in transfers:
        if int(record['bytes']) != KNOWN['constants.bin']:
            continue
        data = transfer_data(record, dumps)
        digest = hashlib.sha256(data).hexdigest()
        if not expected or digest == expected['constants.bin']:
            constants[digest] = data
    if len(constants) != 1:
        raise ValueError('No unique complete constants transfer matches the requested model')
    return {'weights.bin': next(iter(weights.values())), 'constants.bin': next(iter(constants.values()))}


def abi_module():
    specification = importlib.util.spec_from_file_location('netal_sr_abi', Path(__file__).with_name('netal_sr_abi.py'))
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def publish(model, output, receipt, abi=None):
    """Publish both files together into a new directory; never overwrite an installed model."""
    if output.exists() or output.is_symlink():
        raise ValueError('Output already exists; choose a new directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.netal-model-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'model'
        stage.mkdir()
        for name in KNOWN:
            (stage / name).write_bytes(model[name])
        if abi is not None:
            templates, manifest = abi
            (stage / 'abi').mkdir()
            for name, data in templates.items():
                if name not in abi_module().CONTRACTS: raise ValueError('Unsupported boundary file')
                (stage / 'abi' / name).write_bytes(data)
            (stage / 'abi/manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (stage / 'extraction-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        # Recheck immediately before publication; normal installations are serialized by the installer.
        if output.exists() or output.is_symlink():
            raise ValueError('Output appeared during extraction; refusing replacement')
        stage.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dll', type=Path)
    parser.add_argument('--capture-dir', type=Path)
    parser.add_argument('--harness', type=Path, help='legacy developer override')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--scenario', default='quality')
    parser.add_argument('--work', type=Path, help='new directory to retain this run for diagnosis')
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--allow-unknown-model', action='store_true', help='research only: skip known model hashes; never reports verified')
    args = parser.parse_args()
    try:
        dll, output = args.dll.resolve(strict=True), args.out.absolute()
        if output.exists() or output.is_symlink():
            raise ValueError('Output already exists; choose a new directory')
        if args.timeout <= 0:
            raise ValueError('Timeout must be positive')
        dll_sha = sha(dll)
        known = dll_sha == REFERENCE['source_dll_sha256'] and args.scenario == REFERENCE['scenario']
        if not known and not args.allow_unknown_model:
            raise ValueError('Unsupported DLSS runtime/scenario. Only the hash-pinned quality model is accepted by default.')
        root = Path(__file__).resolve().parents[1]
        capture = args.capture_dir
        if capture is None and args.harness:
            capture = args.harness / 'build/dlss_sr_capture'
        if capture is None:
            capture = root / 'build/model-capture'
            if not capture.exists():
                capture = root / 'vendor/dlss-capture'
        capture = capture.resolve()
        # TemporaryDirectory isolates concurrent callers; explicit diagnostic work dirs must be new.
        temporary = tempfile.TemporaryDirectory(prefix='netal-dlss-') if args.work is None else None
        work = Path(temporary.name) if temporary else args.work.absolute()
        if not temporary:
            work.mkdir(parents=True, exist_ok=False)
        try:
            print('extracting: running the local DLL loader under Wine', flush=True)
            trace, dumps = run_loader(dll, capture, work, args.scenario, args.timeout)
            allocations, transfers = arenas(trace)
            model = select_model(allocations, transfers, dumps, REFERENCE['files'] if known else None)
            abi = abi_module().extract(trace, dumps) if known else None
            receipt = {
                'schema': 'netal.sr.extraction.v1', 'source_dll_sha256': dll_sha,
                'scenario': args.scenario, 'model': REFERENCE['model'] if known else 'unknown',
                'matches_reference_model_bytes': known,
                'inference_validated': False, 'gameplay_validated': False,
                'capture_helpers': {name: sha(capture / name) for name in CAPTURE_FILES},
                'trace_sha256': sha(trace),
                'files': {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()} for name, data in model.items()},
            }
            if abi is not None: receipt['boundary_abi'] = abi[1]
            publish(model, output, receipt, abi)
        finally:
            if temporary:
                temporary.cleanup()
        print('result=pass model_bytes=' + ('reference-matched' if known else 'unverified'))
        return 0
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print('result=fail ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
