#!/usr/bin/env python3
"""Exercise assembly failures using caller-supplied engine and runtime fixtures.

Copies inputs into a temporary directory. Does not execute Metal or modify fixtures.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

spec = importlib.util.spec_from_file_location('assembly', Path(__file__).with_name('assemble-netal-fg-runtime.py'))
assembly = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assembly)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    args = parser.parse_args()
    checks = []
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        engine, inputs = root / 'engine', root / 'inputs'
        shutil.copytree(args.engine, engine)
        shutil.copytree(args.inputs / 'model', inputs / 'model')
        for name in ('rom.bin', 'sigmoid.bin', 'ex2-unit.bin', 'ex2-fraction.bin'):
            shutil.copyfile(args.inputs / name, inputs / name)
        math = {name: inputs / name for name in assembly.load_json(assembly.CONTRACT)['math']}
        model = inputs / 'model'
        output = root / 'bundle'
        manifest = assembly.assemble(engine, model, math, output)
        assert manifest['execution_validated'] is False
        assert assembly.file_set(output) == set(manifest['files']) | {'manifest.json'}
        for name, expected in manifest['files'].items():
            assembly.checked_file(output / name, expected)
        checks.append('successful assembly and complete inventory')
        preserved = (output / 'manifest.json').read_bytes()

        def reject(label, target=None):
            target = target or root / ('rejected-' + str(len(checks)))
            existed = target.exists()
            try:
                assembly.assemble(engine, model, math, target)
            except (ValueError, OSError):
                if not existed:
                    assert not target.exists(), 'partial output published'
                assert (output / 'manifest.json').read_bytes() == preserved
                checks.append(label)
            else:
                raise AssertionError('Accepted ' + label)

        reject('existing output preserved', output)
        reject('output inside model rejected', model / 'nested')
        reject('output inside engine rejected', engine / 'nested')
        extra = model / 'unexpected.bin'
        extra.write_bytes(b'extra')
        reject('extra model buffer rejected')
        extra.unlink()
        token = next(model.glob('*.bin'))
        saved = token.read_bytes()
        token.unlink()
        reject('missing model buffer rejected')
        token.symlink_to(root / 'absent')
        reject('model symlink rejected')
        token.unlink()
        token.write_bytes(bytes([saved[0] ^ 1]) + saved[1:])
        reject('same-size model corruption rejected')
        token.write_bytes(saved)
        table = math['sigmoid.bin']
        saved = table.read_bytes()
        table.write_bytes(saved[:-1])
        reject('truncated math table rejected')
        table.write_bytes(saved)
        library = next((engine / 'libs').glob('*.metallib'))
        saved = library.read_bytes()
        library.write_bytes(bytes([saved[0] ^ 1]) + saved[1:])
        reject('engine library corruption rejected')
        library.write_bytes(saved)
        extra = engine / 'extra.bin'
        extra.write_bytes(b'extra')
        reject('unlisted engine file rejected')
        extra.unlink()
        receipt = engine / 'build-receipt.json'
        saved = receipt.read_bytes()
        value = json.loads(saved)
        name = next(iter(value['files']))
        value['files']['../escape'] = value['files'].pop(name)
        receipt.write_text(json.dumps(value))
        reject('unsafe inventory path rejected')
        receipt.write_text('{"schema": "first", "schema": "second"}')
        reject('duplicate manifest key rejected')
        receipt.write_bytes(saved)
        value = json.loads(saved)
        hardware = next((engine / 'libs').glob('*_hw.metallib'))
        hardware_bytes = hardware.read_bytes()
        hardware.unlink()
        del value['files']['libs/' + hardware.name]
        value['kernels'] = [k for k in value['kernels'] if k['library'] != hardware.name]
        receipt.write_text(json.dumps(value))
        reject('missing hardware variant rejected even with consistent inventory')
        hardware.write_bytes(hardware_bytes)
        receipt.write_bytes(saved)
        header = engine / 'include/fg_runtime.h'
        saved = header.read_bytes()
        header.unlink()
        original = root / 'original-header'
        original.write_bytes(saved)
        header.symlink_to(original)
        reject('engine symlink rejected')
        header.unlink()
        header.write_bytes(saved)
        assert not list(root.glob('.netal-fg-assembly-*')), 'staging directory leaked'
        checks.append('no staging residue')
    print(json.dumps({'status': 'pass', 'checks': checks, 'gpu_executed': False}, indent=2))


if __name__ == '__main__':
    main()
