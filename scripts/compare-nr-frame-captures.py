#!/usr/bin/env python3
"""Post-execution byte comparison only. Never imports or invokes NR execution.

Equality can reproduce a known failure; it is NOT visual or parity acceptance.
Exit 0 means evidence was valid and compared, even when bytes differ.
"""
import argparse
import hashlib
import json
import pathlib
import numpy as np

PLANES = {'game-input.bin', 'model-input.rgba16f',
          'model-output.rgba16f', 'game-output.bin'}


def read_capture(path):
    path = pathlib.Path(path)
    raw_manifest = path.read_bytes()
    record = json.loads(raw_manifest)
    if record.get('schema') != 'netal.nr.live-frame-capture.v1':
        raise ValueError('unsupported capture schema')
    if record.get('gpu_completed') is not True:
        raise ValueError('capture GPU completion not established')
    files = record.get('files')
    if not isinstance(files, list) or len(files) != len(PLANES):
        raise ValueError('capture must contain all four planes')
    entries = {}
    for e in files:
        name = e.get('file')
        if name not in PLANES or name in entries:
            raise ValueError('unknown/duplicate capture plane')
        width, height, bpp, fmt, size = (e.get(k) for k in
            ('width', 'height', 'bytes_per_pixel', 'metal_pixel_format', 'bytes'))
        if any(type(v) is not int for v in (width, height, bpp, fmt, size)):
            raise ValueError('non-integer plane metadata')
        if not (0 < width <= 1920 and 0 < height <= 1200):
            raise ValueError('plane extent outside diagnostic bound')
        if (fmt, bpp) not in ((92, 4), (115, 8)) or size != width*height*bpp:
            raise ValueError('plane format/size mismatch')
        plane = path.parent/name
        if plane.stat().st_size != size:
            raise ValueError('plane file size mismatch')
        raw = plane.read_bytes()
        if hashlib.sha256(raw).hexdigest() != e.get('sha256'):
            raise ValueError('plane SHA256 mismatch')
        entries[name] = (e, raw)
    return record, entries, hashlib.sha256(raw_manifest).hexdigest()


def compare(left, right):
    lm, le, lh = read_capture(left)
    rm, re, rh = read_capture(right)
    planes = {}
    for name in sorted(PLANES):
        a, ab = le[name]
        b, bb = re[name]
        geometry_equal = all(a[k] == b[k] for k in
            ('width', 'height', 'metal_pixel_format', 'bytes_per_pixel', 'bytes'))
        equal = geometry_equal and ab == bb
        detail = {'geometry_equal': geometry_equal, 'bytes_equal': equal,
                  'left_sha256': a['sha256'], 'right_sha256': b['sha256']}
        if geometry_equal:
            diff = np.flatnonzero(np.frombuffer(ab, dtype=np.uint8) !=
                                  np.frombuffer(bb, dtype=np.uint8))
            detail['mismatching_bytes'] = int(diff.size)
            detail['first_mismatching_byte'] = int(diff[0]) if diff.size else None
        planes[name] = detail
    comparable = all(k in lm and k in rm and lm[k] == rm[k] for k in
        ('model_identity', 'arithmetic', 'reset_only', 'game_extent',
         'model_extent', 'model_content'))
    # The original v1 capture schema implicitly used raw color/identity export.
    comparable = comparable and lm.get('color_mode','raw') == rm.get('color_mode','raw')
    comparable = comparable and lm.get('color_reconstruction','identity') == rm.get('color_reconstruction','identity')
    return {'schema': 'netal.nr.post-execution-capture-comparison.v1',
            'left_manifest_sha256': lh, 'right_manifest_sha256': rh,
            'execution_contract_metadata_equal': comparable,
            'all_plane_bytes_equal': all(v['bytes_equal'] for v in planes.values()),
            'planes': planes, 'visual_acceptance': False,
            'rtx_parity_acceptance': False, 'live_game_acceptance': False,
            'note': 'Comparison only. Identical output can reproduce a failure; '
                    'no expected bytes are fed into execution.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('left_manifest')
    ap.add_argument('right_manifest')
    args = ap.parse_args()
    print(json.dumps(compare(args.left_manifest, args.right_manifest), indent=2))


if __name__ == '__main__':
    main()
