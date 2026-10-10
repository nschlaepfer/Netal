#!/usr/bin/env python3
"""Fault tests for model reconstruction/publication; synthetic data, no DLL or Wine."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile

path = Path(__file__).resolve().parents[1] / 'tools/netal-dlss-extract.py'
spec = importlib.util.spec_from_file_location('extract', path)
extract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract)
checks = 0


def check(condition, name):
    global checks
    assert condition, name
    checks += 1
    print('ok ' + name)


def fails(operation, name):
    try:
        operation()
    except ValueError:
        check(True, name)
    else:
        check(False, name)


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    def transfer(identity, offset, data, length=None):
        (root / ('transfer-%04d.bin' % identity)).write_bytes(data)
        return {'destination': hex(0x1000 + offset), 'bytes': len(data) if length is None else length, 'transfer_id': identity}
    a = transfer(1, 4, b'EFGH')
    b = transfer(2, 0, b'ABCD')
    check(extract.rebuild(0x1000, 8, [a, b], root) == b'ABCDEFGH', 'uses destination offsets, not recording order')
    check(extract.rebuild(0x1000, 8, [a], root) is None, 'holes are not silently zero-filled')
    bad = transfer(3, 6, b'WXYZ')
    fails(lambda: extract.rebuild(0x1000, 8, [bad], root), 'rejects transfer past arena end')
    short = transfer(4, 0, b'AB', length=8)
    fails(lambda: extract.rebuild(0x1000, 8, [short], root), 'rejects truncated transfer')
    fails(lambda: extract.rebuild(0x1000, extract.MAX_ARENA_BYTES + 1, [], root), 'rejects unbounded allocation')
    check(extract.rebuild(0x1000, 8, [b, a, transfer(5, 4, b'IJKL')], root) == b'ABCDIJKL', 'repeated upload preserves final write order')
    model = {'weights.bin': b'weights', 'constants.bin': b'constants'}
    output = root / 'published'
    extract.publish(model, output, {'test': True})
    check((output / 'weights.bin').read_bytes() == b'weights' and (output / 'constants.bin').read_bytes() == b'constants', 'publishes both model files')
    fails(lambda: extract.publish(model, output, {}), 'refuses existing output')
    check((output / 'weights.bin').read_bytes() == b'weights', 'existing output survives refusal')
    try:
        extract.publish({'weights.bin': b'incomplete'}, root / 'partial', {})
    except KeyError:
        pass
    check(not (root / 'partial').exists(), 'failed preparation exposes no partial model')
    # An unrelated larger arena cannot win simply because it is larger.
    expected_weights = b'w' * extract.KNOWN['weights.bin']
    expected_constants = b'c' * extract.KNOWN['constants.bin']
    expected = {name: hashlib.sha256(data).hexdigest() for name, data in [('weights.bin', expected_weights), ('constants.bin', expected_constants)]}
    transfers = [transfer(6, 0, expected_weights), transfer(7, 0x1000000, expected_constants)]
    allocations = [{'pointer': hex(0x1000), 'bytes': len(expected_weights)}, {'pointer': hex(0x1000 + 0x1000000), 'bytes': len(expected_constants)}]
    selected = extract.select_model(allocations, transfers, root, expected)
    check(selected['weights.bin'] == expected_weights and selected['constants.bin'] == expected_constants, 'selects exact reference bytes')
    wrong = dict(expected, **{'weights.bin': '0' * 64})
    fails(lambda: extract.select_model(allocations, transfers, root, wrong), 'fails when complete data has wrong identity')
print(f'{checks} model extraction checks passed')
