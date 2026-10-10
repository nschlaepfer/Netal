#!/usr/bin/env python3
"""Verify rebuilt loaded fixture constants against an immutable NR audit ledger.

Does not open unlisted archived activation fixtures or execute the model.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = '/private/tmp/claude-501/-Users-nico/deb7f1d0-b1d5-4951-9b30-84ff130544fc/scratchpad/nrwt/'
PARENT = '5bd81e7b5094114f3a1efd55a1a952f9abfaea1d3e2aaacda46e1a3326d92a50'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def audit(peer, decoder_stage=None):
    source = ROOT / 'build/nr_resident_review/live-v17-black3-v2.json'
    data = json.loads(source.read_text())
    need(data.pop('receipt_root_sha256') == PARENT and canonical(data) == PARENT, 'historical ledger identity')
    records = []
    for name, expected in data['artifacts'].items():
        if name.startswith(OLD) and '/fixtures/' in name:
            path = peer / name[len(OLD):]
            observed = sha(path) if path.is_file() else None
            records.append({'historical_path': name, 'current_path': str(path),
                            'expected_sha256': expected, 'observed_sha256': observed,
                            'bytes': path.stat().st_size if path.is_file() else None,
                            'matches': observed == expected})
    need(len(records) == 100, 'complete historically loaded fixture inventory')
    if decoder_stage is not None:
        names = {f'{kind}-{n}.bin' for kind in ('params', 'weights') for n in range(133, 157)}
        need({p.name for p in decoder_stage.iterdir()} == names, 'stage-only directory must contain exactly 48 parameter/weight files')
        stages = [(name, expected) for name, expected in data['artifacts'].items() if '/stage/' in name]
        need(len(stages) == 48 and {Path(n).name for n, _ in stages} == names, 'historical decoder stage inventory')
        for name, expected in stages:
            path = decoder_stage / Path(name).name
            observed = sha(path)
            records.append({'historical_path': name, 'current_path': str(path),
                            'expected_sha256': expected, 'observed_sha256': observed,
                            'bytes': path.stat().st_size, 'matches': observed == expected})
    # Verify the observation did not race a still-running regeneration.
    for row in records:
        if row['observed_sha256'] is not None:
            need(sha(Path(row['current_path'])) == row['observed_sha256'], 'fixture changed during review')
    return {'schema': 'neuralstudio.nr.recovered-fixtures.v1', 'parent_root': PARENT,
            'result': 'matching' if all(r['matches'] for r in records) else 'incomplete_or_mismatched',
            'checked': len(records), 'matching': sum(r['matches'] for r in records), 'artifacts': records,
            'scope': 'historically loaded fixture constants' + (' plus 48 decoder parameter/weight files' if decoder_stage is not None else ''),
            'live_model_executed': False, 'runtime_changed': False, 'full_model_parity': False,
            'verifier_sha256': sha(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--peer-root', type=Path, required=True)
    parser.add_argument('--decoder-stage', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    need(not args.out.exists(), 'preserve existing report')
    result = audit(args.peer_root.resolve(), args.decoder_stage.resolve() if args.decoder_stage is not None else None)
    result['receipt_root_sha256'] = canonical(result)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}, indent=2))
    return 0 if result['result'] == 'matching' else 1


if __name__ == '__main__':
    raise SystemExit(main())
