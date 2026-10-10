#!/usr/bin/env python3
"""Check an exported source tree against policy and its complete SHA-256 inventory."""
import argparse
import hashlib
import json
from pathlib import Path
from public_source_policy import allowed, content_findings


def audit(root):
    failures = []
    try:
        manifest = json.loads((root / 'SOURCE_MANIFEST.json').read_text())
        entries = manifest['files']
        expected = {item['path']: item for item in entries}
        if manifest.get('git_history_included') is not False or len(entries) != len(expected):
            failures.append({'path': 'SOURCE_MANIFEST.json', 'reason': 'invalid-inventory'})
    except (OSError, ValueError, KeyError, TypeError):
        return [{'path': 'SOURCE_MANIFEST.json', 'reason': 'missing-or-invalid-inventory'}]
    seen = set()
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            failures.append({'path': relative, 'reason': 'symlink'})
            continue
        if path.is_dir():
            if path.name == '.git':
                failures.append({'path': relative, 'reason': 'git-history'})
            continue
        if relative == 'SOURCE_MANIFEST.json':
            continue
        seen.add(relative)
        data = path.read_bytes()
        if not allowed(relative):
            failures.append({'path': relative, 'reason': 'outside-source-policy'})
        for finding in content_findings(data, path.suffix == '.png', relative):
            failures.append({'path': relative, 'reason': finding})
        entry = expected.get(relative)
        if not entry or entry.get('sha256') != hashlib.sha256(data).hexdigest() or entry.get('bytes') != len(data):
            failures.append({'path': relative, 'reason': 'inventory-mismatch'})
    for missing in sorted(set(expected) - seen):
        failures.append({'path': missing, 'reason': 'inventory-file-missing'})
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    failures = audit(args.directory)
    print(json.dumps({'status': 'fail' if failures else 'pass', 'scope': 'content-policy-and-inventory-only', 'publication_clearance': False, 'findings': failures}, indent=2))
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
