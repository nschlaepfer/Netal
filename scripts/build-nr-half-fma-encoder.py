#!/usr/bin/env python3
"""Isolated live-validation object: replace only the candidate C32 prefix library."""
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


def sealed(path, root):
    r = json.loads(path.read_text()); b = dict(r); h = b.pop('receipt_root_sha256')
    assert canonical(b) == h == root
    return r


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); p.add_argument('--spatial', action='store_true', help='Retain the already accepted uniform/spatial motion admission and fitted sampler'); a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    parent = repo / ('build/nr_encoder_temporal/spatial-v1' if a.spatial else 'build/nr_encoder_temporal/build-v2-sass')
    r = sealed(parent / 'build.json', '7c5531e9d15b68b4252060bc2e5daeaa55aa62fe155bcb3ec59fe89441ca5844' if a.spatial else 'b2628242a1e425ec460cd2cb5e445da64283d736a3c5b1789bcecfa127c84c9f')
    candidate = repo / 'build/nr_sixteen/half-fma-candidate-v1'
    c = sealed(candidate / 'build.json', 'ad73a1eff565f8885a22ae6ed2f30ae09f9bcfc4d5f871fdacac567537ad296f')
    for group, folder in [('source_files', 'source'), ('assets', 'assets')]:
        for n, h in r[group].items(): assert sha(parent / folder / n) == h
    for n, h in c['files'].items(): assert sha(candidate / n) == h
    assert not a.output.exists(); a.output.mkdir(parents=True)
    shutil.copytree(parent / 'source', a.output / 'source'); shutil.copytree(parent / 'assets', a.output / 'assets')
    prefix = a.output / 'assets/prefix.metallib'
    shutil.copyfile(candidate / 'prefix.metallib', prefix)
    header = a.output / 'source/generated/asset_hashes.h'
    old = header.read_text(); assert old.count(r['assets']['prefix.metallib']) == 1
    header.write_text(old.replace(r['assets']['prefix.metallib'], sha(prefix)))
    source = a.output / 'source'
    obj = a.output / 'nr_encoder_runtime.o'
    command = ['xcrun', '-sdk', 'macosx', 'clang++', '-std=c++20', '-O2', '-fobjc-arc', '-Wall', '-Wextra', '-Werror', '-Wno-deprecated-declarations', '-c', str(source / 'nr_encoder_runtime.mm'), '-o', str(obj)]
    with (a.output / 'compile.log').open('x') as log: run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    assets = {n: sha(a.output / 'assets' / n) for n in r['assets']}
    assert [n for n in assets if assets[n] != r['assets'][n]] == ['prefix.metallib']
    result = {'schema': 'neuralstudio.nr.half-fma-encoder-build.v1', 'result': 'compiled_candidate' if run.returncode == 0 else 'failed',
              'parent_root': r['receipt_root_sha256'], 'candidate_library_root': c['receipt_root_sha256'],
              'assets': assets, 'source_files': {str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()},
              'command': command, 'exit_code': run.returncode, 'accepted': False, 'production_changed': False,
              'change': 'C32 normalization native half FMA instead of FP32 FMA then half, same fusion order; all other assets/host logic unchanged.'}
    if run.returncode == 0: result['object_sha256'] = sha(obj)
    result['receipt_root_sha256'] = canonical(result)
    (a.output / 'build.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('result', 'receipt_root_sha256', 'object_sha256') if k in result}))
    if run.returncode: raise SystemExit(3)


if __name__ == '__main__': main()
