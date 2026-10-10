#!/usr/bin/env python3
"""Check local Metal EX2 baselines before using the pinned FG correction data.

Compiles a scalar probe and runs two fresh GPU processes, without model inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from netal_fg_calibration import BASELINES, load

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_run(directory, poison):
    receipt = json.loads((directory / 'run-receipt.json').read_text())
    if (receipt.get('result') != 'pass' or receipt.get('canary_mismatched_bytes') != 0
            or receipt.get('poison_byte') != poison):
        raise ValueError('Scalar probe failed or poison/guard checks are missing')
    for name, expected in BASELINES.items():
        path = directory / f'fast-{name}-u32.bin'
        if (path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']
                or receipt.get(name + '_records') != expected['records']
                or receipt.get(name + '_sha256') != expected['sha256']):
            raise ValueError('Unsupported native EX2 baseline: ' + name)
    return receipt


def check(output, probe_override=None):
    load(ROOT)
    if output.exists() or output.is_symlink():
        raise ValueError('Output exists; use a new directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    compiler = Path(subprocess.check_output(['xcrun', '--find', 'metal'], text=True).strip())
    linker = compiler.with_name('metallib')
    if not linker.is_file():
        linker = Path(subprocess.check_output(['xcrun', '--find', 'metallib'], text=True).strip())
    with tempfile.TemporaryDirectory(prefix='.fg-calibration-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'check'
        stage.mkdir()
        library = stage / 'probe.metallib'
        if probe_override is None:
            subprocess.run([str(compiler), '-std=metal3.1', '-c',
                            str(ROOT / 'src/fg_math_ex2/fast_table_probe.metal'),
                            '-o', str(stage / 'probe.air')], check=True)
            subprocess.run([str(linker), str(stage / 'probe.air'), '-o', str(library)], check=True)
        else:
            library = probe_override.resolve()
        host = stage / 'probe-host'
        subprocess.run([
            'xcrun', 'clang++', '-std=c++20', '-fobjc-arc', '-Wall', '-Wextra', '-Werror',
            '-Wno-deprecated-declarations', str(ROOT / 'src/fg_math_ex2/macos_fast_table_probe.mm'),
            '-framework', 'Foundation', '-framework', 'Metal', '-framework', 'Security', '-o', str(host),
        ], check=True)
        runs = []
        for poison in (0xa5, 0x5a):
            run = stage / f'{poison:02x}'
            completed = subprocess.run([str(host), str(library), str(run), f'{poison:02x}', f'{poison:02x}'],
                                       capture_output=True, text=True, check=True)
            (stage / f'{poison:02x}.log').write_text(completed.stdout + completed.stderr)
            runs.append(verify_run(run, poison))
        if len({run['process_id'] for run in runs}) != 2:
            raise ValueError('Probe processes were not distinct')
        receipt = {
            'schema': 'netal.fg.scalar-compatibility.v1', 'status': 'pass',
            'model_inputs': 0, 'native_ex2_baselines': BASELINES, 'fresh_process_count': 2,
            'probe_library_sha256': sha(library), 'probe_host_sha256': sha(host),
            'metal_version': subprocess.check_output([str(compiler), '--version'], text=True).splitlines()[0],
            'runs': runs, 'gameplay_validated': False, 'redistribution_clearance': False,
        }
        (stage / 'compatibility.json').write_text(json.dumps(receipt, indent=2) + '\n')
        if output.exists() or output.is_symlink():
            raise ValueError('Output appeared during check; preserved')
        stage.rename(output)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--probe-metallib', type=Path,
                        help='diagnostic probe override; the same pinned baseline checks still apply')
    args = parser.parse_args()
    try:
        check(args.output.absolute(), args.probe_metallib)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'calibration check failed: {error}\n')
    print('result=pass fresh_processes=2 scalar_inputs_per_process=25165826 model_inputs=0')


if __name__ == '__main__':
    main()
