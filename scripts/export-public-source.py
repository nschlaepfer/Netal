#!/usr/bin/env python3
"""Export reviewable source without history, models or compiled runtime code.

Five explicitly pinned standalone scalar-calibration data files are allowed by policy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from public_source_policy import allowed, content_findings


def export(root, output):
    root = root.resolve()
    output = output.resolve()
    report = output.with_name(output.name + "-exclusions.json")
    if report.exists():
        raise ValueError("exclusion report already exists")
    if output.exists() or output.is_symlink():
        raise ValueError('destination must not exist')
    try:
        output.relative_to(root)
        raise ValueError('destination must be outside the source checkout')
    except ValueError as error:
        if str(error) == 'destination must be outside the source checkout':
            raise
    names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=root)
    paths = sorted(set(x.decode('utf-8') for x in names.split(b'\0') if x))
    selected, excluded, blockers = [], [], []
    for relative in paths:
        source = root / relative
        if not allowed(relative):
            excluded.append({'path': relative, 'reason': 'outside-public-source-policy'})
            continue
        if source.is_symlink() or not source.is_file():
            blockers.append({'path': relative, 'reasons': ['symlink-or-missing-source']})
            continue
        data = source.read_bytes()
        findings = content_findings(data, source.suffix == '.png', relative)
        # Preserve machine-specific helpers in the research archive, never rewrite their code.
        if findings == ['developer-home-path']:
            excluded.append({'path': relative, 'reason': 'machine-specific-research-source'})
            continue
        if findings:
            blockers.append({'path': relative, 'reasons': findings})
        else:
            selected.append((relative, source, data))
    if blockers:
        # Never include matched secret values in output.
        raise ValueError(json.dumps({'blocked': blockers}, indent=2))
    output.mkdir(parents=True)
    records = []
    for relative, source, data in selected:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        mode = 0o755 if source.stat().st_mode & 0o111 else 0o644
        target.chmod(mode)
        records.append({'path': relative, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data), 'mode': oct(mode)})
    base = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    manifest = {'schema': 'netal.public-source.v1', 'base_revision': base,
                'working_changes_included': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root).strip()), 'git_history_included': False,
                'status': 'review-candidate', 'files': records}
    (output / 'SOURCE_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    # Exclusions belong beside the public source, not inside it: they name private research files.
    report.write_text(json.dumps({'included': len(records), 'excluded': excluded}, indent=2) + '\n')
    return len(records), len(excluded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        included, excluded = export(Path(__file__).resolve().parents[1], args.output)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(f'exported={included} excluded={excluded} status=review-candidate')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
