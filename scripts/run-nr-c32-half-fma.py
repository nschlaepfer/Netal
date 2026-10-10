#!/usr/bin/env python3
"""Four fresh half-FMA candidate replays; expected data is post-run comparison only."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def canonical(r):
    return hashlib.sha256(json.dumps(r, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    repo = Path(__file__).resolve().parents[1]
    out = Path(sys.argv[1]).resolve()
    assert not out.exists()
    out.mkdir(parents=True)
    root = repo / 'build/nr_sixteen'
    source = root / 'early-comparison-v1'
    host = root / 'window-probe-v1/probe'
    build = repo / 'build/nr_encoder_temporal/build-v2-sass'
    manifest = json.loads((build / 'build.json').read_text())
    assets = build / 'assets'
    library, weight = assets / 'prefix.metallib', assets / 'weights/block4.bin'
    assert sha(library) == manifest['assets']['prefix.metallib']
    candidate = root / 'half-fma-candidate-v1'
    cr = json.loads((candidate / 'build.json').read_text())
    cb = dict(cr)
    ch = cb.pop('receipt_root_sha256')
    assert canonical(cb) == ch == 'ad73a1eff565f8885a22ae6ed2f30ae09f9bcfc4d5f871fdacac567537ad296f'
    for n, h in cr['files'].items():
        assert sha(candidate / n) == h
    library = candidate / 'prefix.metallib'
    assert sha(weight) == manifest['assets']['weights/block4.bin']
    assert sha(host) == '9ff9ad10b7c6a0ca6a92cf5f4350c2c7f8b03762daa8bdbadaffa014c48ebef5'
    inp = source / 'input.e4m3'
    assert sha(inp) == '523e57ef59591bfcd471aa9f4ca83e2f41dc0d70fc56076aebc20b2146b36a5e'
    pins = {str(p): sha(p) for p in (inp, host, library, weight)}
    runs = []
    for label in ('a5-a', 'a5-b', '5a-a', '5a-b'):
        dest = out / label
        cmd = [str(host), str(library), str(inp), str(weight), label[:2], str(dest)]
        with (out / (label + '.stdout')).open('x') as stdout, (out / (label + '.stderr')).open('x') as stderr:
            run = subprocess.run(cmd, stdout=stdout, stderr=stderr)
        assert run.returncode == 0, 'preserve failed diagnostic'
        r = json.loads((dest / 'run.json').read_text())
        assert r['archived_component_input'] and not r['full_model_inference'] and not r['expected_file_opened']
        assert r['coverage_errors'] == r['canary_errors'] == r['immutable_errors'] == 0
        actual = (dest / 'output.e4m3').read_bytes()
        expected = (source / 'expected-comparison-only.e4m3').read_bytes()
        previous = (source / 'metal-comparison-only.e4m3').read_bytes()
        assert len(actual) == len(expected) == len(previous) == 2048
        assert actual == expected, 'candidate does not match the RTX comparison window'
        differences = [{'offset': i, 'rtx': e, 'metal': a} for i, (e, a) in enumerate(zip(expected, actual)) if e != a]
        assert len(differences) == 0
        assert sum(a != b for a, b in zip(actual, previous)) == 2
        runs.append({'label': label, 'command': cmd, 'receipt': r, 'differences': differences,
                     'artifacts': {p.name: sha(p) for p in dest.iterdir() if p.is_file()}})
    assert len({r['receipt']['pid'] for r in runs}) == 4
    for path, pin in pins.items():
        assert sha(Path(path)) == pin
    result = {'schema': 'neuralstudio.nr.c32-half-fma-window.v1', 'result': 'pass_bounded_candidate_diagnostic', 'candidate_root': ch, 'accepted_live_inference': False,
              'all_fresh_processes': True, 'all_two_poison_artifacts_equal': all(r['artifacts']['output.e4m3'] == runs[0]['artifacts']['output.e4m3'] for r in runs),
              'full_model_inference': False, 'diagnostic_archived_input': True, 'pins': pins, 'runs': runs,
              'source_sha256': sha(Path(__file__))}
    result['receipt_root_sha256'] = canonical(result)
    (out / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'root': result['receipt_root_sha256'], 'differences': runs[0]['differences']}))


if __name__ == '__main__':
    main()
