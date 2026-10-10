#!/usr/bin/env python3
"""Exercise atomic installation and interruption behavior using synthetic model bytes."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile

path = Path(__file__).resolve().parents[1] / 'tools/netal-model-install.py'
spec = importlib.util.spec_from_file_location('installer', path)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)
checks = 0


def check(condition, message):
    global checks
    assert condition, message
    checks += 1
    print('ok ' + message)


def fail(operation, message):
    try:
        operation()
    except (ValueError, OSError):
        check(True, message)
    else:
        check(False, message)


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    model = root / 'model'; model.mkdir()
    contents = {'weights.bin': b'new-weights', 'constants.bin': b'new-constants'}
    for name, data in contents.items():
        (model / name).write_bytes(data)
    installer.extract.KNOWN = {name: len(data) for name, data in contents.items()}
    installer.extract.REFERENCE = {'model': 'test-only', 'source_dll_sha256': 'test-source',
                                    'files': {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}}
    receipt = {'source_dll_sha256': 'test-source', 'matches_reference_model_bytes': True}
    def bundle(name):
        target = root / name; target.mkdir()
        (target / 'model').mkdir()
        (target / 'model/weights.bin').write_bytes(b'old-weights')
        (target / 'engine.txt').write_text('unchanged-engine')
        (target / 'manifest.json').write_text(json.dumps({'model': 'test-only', 'files': []}))
        return target
    def snapshot(target):
        return {p.relative_to(target).as_posix(): p.read_bytes() for p in target.rglob('*') if p.is_file()}

    target = bundle('runtime')
    before = snapshot(target)
    installer.install(model, target, receipt, dry_run=True)
    check(snapshot(target) == before, 'dry run leaves installed tree unchanged')
    swap = installer.atomic_exchange
    def reject(*args):
        raise OSError('simulated filesystem refusal')
    installer.atomic_exchange = reject
    fail(lambda: installer.install(model, target, receipt), 'swap refusal is reported')
    check(snapshot(target) == before, 'failed commit leaves old model and manifest intact')
    installer.atomic_exchange = swap
    with installer.install_lock(target):
        def another():
            with installer.install_lock(target):
                pass
        fail(another, 'concurrent installer is refused')
    with installer.install_lock(target):
        installer.install(model, target, receipt)
    check((target / 'model/weights.bin').read_bytes() == contents['weights.bin'], 'new weights installed')
    check((target / 'model/constants.bin').read_bytes() == contents['constants.bin'], 'new constants installed')
    check((target / 'engine.txt').read_text() == 'unchanged-engine', 'engine assets preserved')
    manifest = json.loads((target / 'manifest.json').read_text())
    check(len(manifest['files']) == 2 and manifest['ingest']['matches_reference_model_bytes'], 'manifest and model agree')
    check(not manifest['ingest']['inference_validated_by_install'], 'model installation does not claim inference validation')
    previous = snapshot(target)
    (model / 'weights.bin').write_bytes(b'wrong-weights')
    fail(lambda: installer.install(model, target, receipt), 'modified model is refused')
    check(snapshot(target) == previous, 'invalid model cannot alter the installed bundle')
    (model / 'weights.bin').write_bytes(contents['weights.bin'])
    linked = root / 'linked'; linked.symlink_to(target, target_is_directory=True)
    fail(lambda: installer.install(model, linked, receipt), 'linked target is refused')

    # ABI metadata is auxiliary to the engine. Updating it must not discard user sidecars.
    abi_input = model / 'abi'; abi_input.mkdir()
    abi_target = target / 'model/abi'; abi_target.mkdir()
    (abi_target / 'notes.txt').write_text('keep this local note')
    original_loader = installer.extract.abi_module
    class SyntheticABI:
        @staticmethod
        def validate_package(path):
            return {'template.bin': b'synthetic-template'}, {'schema': 'test-only'}
    installer.extract.abi_module = lambda: SyntheticABI
    installer.install(model, target, receipt)
    check((abi_target / 'notes.txt').read_text() == 'keep this local note', 'ABI update preserves unrelated local sidecars')
    check((abi_target / 'template.bin').read_bytes() == b'synthetic-template', 'ABI metadata installed in the model package')
    installer.extract.abi_module = original_loader
    abi_input.rmdir()

    # Fork without threads: force process death immediately after the atomic exchange. The
    # installation must already be complete even though no normal cleanup/report can run.
    interrupted = bundle('interrupted')
    child = os.fork()
    if child == 0:
        def stop_after_exchange(first, second):
            swap(first, second)
            os._exit(77)
        installer.atomic_exchange = stop_after_exchange
        installer.install(model, interrupted, receipt)
        os._exit(78)
    _, status = os.waitpid(child, 0)
    check(os.waitstatus_to_exitcode(status) == 77, 'interruption occurred directly after commit')
    check((interrupted / 'model/weights.bin').read_bytes() == contents['weights.bin'] and
          (interrupted / 'model/constants.bin').read_bytes() == contents['constants.bin'], 'interrupted process leaves a complete new model')
    check(json.loads((interrupted / 'manifest.json').read_text())['ingest']['matches_reference_model_bytes'], 'interrupted process leaves the matching manifest')
    check(any(p.name.startswith('.interrupted.install-') for p in root.iterdir()), 'interruption preserves old tree for recovery')
print(f'{checks} model installation checks passed')
