#!/usr/bin/env python3
"""Compare sealed live diagnostics and select an explicitly archived window replay."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def canonical(r):
    return hashlib.sha256(json.dumps(r, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def index(x, y, c):
    window = (y // 8) * 120 + x // 8
    tile = (x % 8 // 4) * 2 + y % 8 // 4
    token = tile * 16 + (y % 4) * 4 + x % 4
    return (window * 64 + token) * 32 + c


def location(i):
    row, c = divmod(int(i), 32)
    window, token = divmod(row, 64)
    tile, pixel = divmod(token, 16)
    x = window % 120 * 8 + tile // 2 * 4 + pixel % 4
    y = window // 120 * 8 + tile % 2 * 4 + pixel // 4
    assert index(x, y, c) == i
    return x, y, c


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    r = json.loads((a.reference / 'reference.json').read_text())
    body = dict(r)
    root = body.pop('receipt_root_sha256')
    assert canonical(body) == root == 'b617baf1b0075fa394f5d0eda097586a6752338b14f7e994d0b967ff887485d5'
    a.output.mkdir(parents=True)
    blocks = []
    first = None
    for b in r['early_blocks']:
        n = b['block']
        rp = a.reference / f'block{n}.e4m3'
        mp = a.native / f'block{n}.e4m3'
        assert sha(rp) == b['canonical_sha256']
        x, y = np.fromfile(rp, dtype=np.uint8), np.fromfile(mp, dtype=np.uint8)
        assert x.size == y.size == b['bytes']
        bad = np.flatnonzero(x != y)
        np.savez_compressed(a.output / f'block{n}-differences.npz', offsets=bad, rtx=x[bad], metal=y[bad])
        samples = [{'offset': int(i), 'xyz': location(i), 'rtx': int(x[i]), 'metal': int(y[i])} for i in bad[:32]]
        windows = sorted({(location(i)[0] // 8, (location(i)[1] + 4) // 8) for i in bad})
        blocks.append({'block': n, 'rtx_sha256': sha(rp), 'metal_sha256': sha(mp),
                       'mismatch_bytes': int(bad.size), 'shifted_block4_windows': windows if n == 4 else [], 'samples': samples})
        if bad.size and first is None:
            first = n
    assert first == 4, 'unexpected frontier; preserve this diagnostic result'
    wx, wy = blocks[3]['shifted_block4_windows'][0]
    source = np.fromfile(a.reference / 'block3.e4m3', dtype=np.uint8)
    expected = np.fromfile(a.reference / 'block4.e4m3', dtype=np.uint8)
    observed = np.fromfile(a.native / 'block4.e4m3', dtype=np.uint8)
    inp, exp, mac, mask = bytearray(), bytearray(), bytearray(), bytearray()
    for tile in range(4):
        for pixel in range(16):
            x = wx * 8 + tile // 2 * 4 + pixel % 4
            y = wy * 8 - 4 + tile % 2 * 4 + pixel // 4
            for c in range(32):
                inside = 0 <= x < 960 and 0 <= y < 576
                i = index(x, y, c) if inside else 0
                inp.append(int(source[i]) if inside else 0)
                exp.append(int(expected[i]) if inside else 0)
                mac.append(int(observed[i]) if inside else 0)
                mask.append(int(inside))
    for name, data in [('input.e4m3', inp), ('expected-comparison-only.e4m3', exp), ('metal-comparison-only.e4m3', mac), ('valid-mask.u8', mask)]:
        (a.output / name).write_bytes(data)
    result = {'schema': 'neuralstudio.nr.frame12-early-comparison.v1', 'result': 'counterexample_localized',
              'reference_root': root, 'first_failing_block': first, 'blocks': blocks, 'selected_shifted_cta': [wx, wy],
              'model_execution': False, 'replay_input_role': 'archived component diagnostic only; not admissible in live full-model inference',
              'source_sha256': sha(Path(__file__)), 'files': {p.name: sha(p) for p in a.output.iterdir()}}
    result['receipt_root_sha256'] = canonical(result)
    (a.output / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
