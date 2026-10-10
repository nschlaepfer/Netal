#!/usr/bin/env python3
"""Check changed Git blobs for restricted payloads and obvious secrets/PII paths.

Reports categories and paths only. This is a contribution guard, not a full
history/privacy/provenance audit. Deleting historical content does not erase it.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from netal_sr_calibration import BINARY_FILES as SR_CALIBRATION
from netal_fg_calibration import BINARY_FILES as FG_CALIBRATION

CALIBRATION = {**SR_CALIBRATION, **FG_CALIBRATION}

DENIED_EXTENSIONS = {'.dll', '.exe', '.so', '.dylib', '.a', '.lib', '.metallib', '.air',
    '.bin', '.cubin', '.ptx', '.safetensors', '.gguf', '.pt', '.pth', '.ckpt', '.p12', '.pfx',
    '.key', '.pem', '.nfg', '.rgba8', '.zip', '.gz', '.7z', '.tar', '.dmp'}
DENIED_ROOTS = {'build', 'work', 'captures', 'checkpoints', '.git'}
PATTERNS = {
    'private-key': re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
    'github-token': re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b'),
    'provider-key': re.compile(rb'\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{40,}\b'),
    'aws-access-key': re.compile(rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'credential-url': re.compile(rb'[a-zA-Z][a-zA-Z0-9+.-]*://[^\s/:@]{2,80}:[^\s/@]{4,160}@'),
    'personal-mac-home': re.compile(b'/' + rb'Users/(?!Shared/|example/|USERNAME/|username/)[A-Za-z0-9_.-]+/'),
    'personal-windows-home': re.compile(rb'(?i)[A-Z]:[\\/]+Users[\\/]+(?!Public[\\/]|example[\\/]|USERNAME[\\/])[A-Za-z0-9_.-]+[\\/]'),
}
MAX_BYTES = 10 * 1024 * 1024


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], input=b'')


def revision(repo, value):
    if not re.fullmatch(r'[0-9a-fA-F]{40}', value):
        raise ValueError('Use a full 40-character commit SHA')
    return git(repo, 'rev-parse', '--verify', value + '^{commit}').decode().strip()


def check(repo, base, head):
    head = revision(repo, head)
    if base == '0' * 40:
        base = git(repo, 'hash-object', '-t', 'tree', '--stdin').decode().strip()
    else:
        base = revision(repo, base)
    raw = git(repo, 'diff', '--name-status', '-z', '--no-renames', base, head).split(b'\0')
    if raw[-1] == b'': raw.pop()
    if len(raw) % 2: raise ValueError('Invalid Git diff format')
    findings = []
    changed = 0
    for status, path_bytes in zip(raw[::2], raw[1::2]):
        if status == b'D': continue
        changed += 1
        path = path_bytes.decode('utf8')
        p = PurePosixPath(path)
        reasons = []
        calibration = CALIBRATION.get(path)
        if p.parts[0] in DENIED_ROOTS or (p.suffix.lower() in DENIED_EXTENSIONS and calibration is None):
            reasons.append('restricted-payload-or-research-output')
        if p.name == '.env' or p.name.startswith('.env.') and p.name != '.env.example':
            reasons.append('environment-secrets-file')
        entry = git(repo, 'ls-tree', '-z', head, '--', path).split(b'\0')[0]
        if not entry or entry.split()[0] not in {b'100644', b'100755'}:
            reasons.append('symlink-submodule-or-nonregular-file')
        size = int(git(repo, 'cat-file', '-s', head + ':' + path))
        if size > MAX_BYTES:
            reasons.append('file-exceeds-10-MiB-review-limit')
        else:
            data = git(repo, 'show', head + ':' + path)
            if calibration is not None:
                if (len(data) != calibration['bytes']
                        or hashlib.sha256(data).hexdigest() != calibration['sha256']):
                    reasons.append('approved-scalar-calibration-identity-mismatch')
            reasons += [name for name, pattern in PATTERNS.items() if pattern.search(data)]
            if data.startswith(b'version https://git-lfs.github.com/spec/'):
                reasons.append('lfs-payload-requires-separate-review')
            if calibration is not None:
                pass  # Only five pinned, model-independent measured scalar assets.
            elif p.suffix.lower() == '.png':
                if not data.startswith(b'\x89PNG\r\n\x1a\n'):
                    reasons.append('invalid-png-header')
            else:
                try: data.decode('utf8')
                except UnicodeDecodeError: reasons.append('unreviewed-binary-content')
                if b'\0' in data: reasons.append('unreviewed-binary-content')
        if reasons: findings.append({'path': path, 'reasons': sorted(set(reasons))})
    return {'status': 'fail' if findings else 'pass', 'scope': 'changed-blobs-only',
            'changed_files_checked': changed, 'findings': findings,
            'matched_values_disclosed': False, 'public_release_clearance': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--base', required=True)
    parser.add_argument('--head', required=True)
    args = parser.parse_args()
    try: result = check(args.repo.resolve(), args.base, args.head)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(2, 'Content check could not inspect the requested revisions: ' + type(error).__name__ + '\n')
    print(json.dumps(result, indent=2))
    return int(result['status'] != 'pass')


if __name__ == '__main__': raise SystemExit(main())
