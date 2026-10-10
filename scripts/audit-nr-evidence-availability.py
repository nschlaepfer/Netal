#!/usr/bin/env python3
"""Check retention of sealed audit inputs; never replace or regenerate evidence.

A valid receipt seal is not proof that its raw evidence remains available.
This is a filesystem inventory, not a new inference or parity acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inspect(receipt_path):
    raw = receipt_path.read_bytes()
    receipt = json.loads(raw)
    claimed = receipt.pop('receipt_root_sha256')
    if canonical(receipt) != claimed:
        raise ValueError('receipt seal mismatch')
    pins = receipt.get('pins')
    if not isinstance(pins, dict) or not pins:
        raise ValueError('no pinned artifact inventory')
    rows = []
    for name, expected in sorted(pins.items()):
        path = Path(name)
        if not path.is_absolute():
            raise ValueError('relative pin has no declared root: ' + name)
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
            raise ValueError('invalid sha256 pin: ' + name)
        row = {'path': name, 'expected_sha256': expected}
        try:
            # stat first distinguishes missing targets from unreadable targets.
            size = path.stat().st_size
            observed = sha(path)
            row.update(status='matching' if observed == expected else 'mismatched',
                       observed_sha256=observed, bytes=size)
        except FileNotFoundError:
            row['status'] = 'missing'
        except OSError as error:
            row.update(status='unreadable', error_type=type(error).__name__)
        rows.append(row)
    if receipt_path.read_bytes() != raw:
        raise ValueError('receipt changed during inventory')
    counts = {state: sum(r['status'] == state for r in rows)
              for state in ('matching', 'missing', 'mismatched', 'unreadable')}
    return {'receipt': str(receipt_path), 'receipt_root_sha256': claimed,
            'receipt_file_sha256': hashlib.sha256(raw).hexdigest(),
            'historical_result': receipt.get('result'), 'counts': counts,
            'all_pins_available_and_matching': counts['matching'] == len(rows),
            'artifacts': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('refusing to overwrite an existing inventory')
    reports = [inspect(p.resolve()) for p in args.receipt]
    result = {'schema': 'neuralstudio.nr.evidence-availability.v1',
              'scope': 'retention inventory only; not parity revalidation',
              'receipts': reports, 'runtime_changed': False,
              'full_goal_complete': False, 'verifier_sha256': sha(Path(__file__))}
    result['inventory_root_sha256'] = canonical(result)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'inventory_root_sha256': result['inventory_root_sha256'],
                      'receipts': [{k: v for k, v in r.items() if k != 'artifacts'}
                                   for r in reports]}, indent=2))
    return 0 if all(r['all_pins_available_and_matching'] for r in reports) else 1


if __name__ == '__main__':
    raise SystemExit(main())
