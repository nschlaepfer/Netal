#!/usr/bin/env python3
"""Isolated normalization-FMA discriminator; does not replace accepted assets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def canonical(r):
    return hashlib.sha256(json.dumps(r, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    bundle = repo / 'build/nr_encoder_fscale/bundle-v1'
    parent = json.loads((bundle / 'manifest.json').read_text())
    b = dict(parent)
    root = b.pop('receipt_root_sha256')
    assert canonical(b) == root == '3350d5f53e391a3be260c2d1d5b60065a9e23e24ba3d9f871f8bf9037ef2d633'
    for n, h in parent['files'].items():
        assert sha(bundle / n) == h
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    src = a.output / 'source'
    shutil.copytree(bundle / 'encoder/source', src)
    target = src / 'swin32_attention.metal'
    code = target.read_text()
    old = '''    half a=half(fma(float(value[8u+p]),float(value[8u+p]),
                    float(half(float(value[24u+p])*float(value[24u+p])))));
    half b=half(fma(float(value[p]),float(value[p]),
                    float(half(float(value[16u+p])*float(value[16u+p])))));'''
    new = '''    half a=fma(value[8u+p],value[8u+p],half(float(value[24u+p])*float(value[24u+p])));
    half b=fma(value[p],value[p],half(float(value[16u+p])*float(value[16u+p])));'''
    assert code.count(old) == 1
    target.write_text(code.replace(old, new))
    metal = subprocess.check_output(['xcrun', '-f', 'metal'], text=True).strip()
    metallib = subprocess.check_output(['xcrun', '-f', 'metallib'], text=True).strip()
    sdk = subprocess.check_output(['xcrun', '--show-sdk-path'], text=True).strip()
    commands = [[metal, '-isysroot', sdk, '-std=metal3.1', '-fno-fast-math', '-ffp-contract=off', '-c', str(src / 'prefix8.metal'), '-o', str(a.output / 'prefix.air')],
                [metallib, str(a.output / 'prefix.air'), '-o', str(a.output / 'prefix.metallib')]]
    with (a.output / 'compile.log').open('x') as log:
        for cmd in commands:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    changed = [p.name for p in src.iterdir() if sha(p) != sha(bundle / 'encoder/source' / p.name)]
    assert changed == ['swin32_attention.metal']
    r = {'schema': 'neuralstudio.nr.c32-half-fma-candidate.v1', 'parent_root': root,
         'change': 'Only two normalization float-FMA-then-half conversions become native half FMA; same fused operand order.',
         'changed_sources': changed, 'commands': commands, 'accepted': False, 'production_changed': False,
         'files': {str(p.relative_to(a.output)): sha(p) for p in a.output.rglob('*') if p.is_file()}}
    r['receipt_root_sha256'] = canonical(r)
    (a.output / 'build.json').write_text(json.dumps(r, indent=2) + '\n')
    print(r['receipt_root_sha256'])


if __name__ == '__main__':
    main()
