#!/usr/bin/env python3
"""Exercise export isolation and detection of accidentally added or modified content."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from public_source_policy import allowed, content_findings


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exporter, auditor = load('export-public-source'), load('audit-public-source')
assert not allowed('work/re9-sr-runtime-bundle/model/weights.bin')
assert not allowed('src/../secrets.txt')
assert not allowed('src/research/evidence/receipt.json')
assert not allowed('runtime/helper.dll')
assert allowed('src/fg_conv/hmma_exact.metalh')
assert allowed('src/fg_runtime/tail_resource_cache.inc')
assert not allowed('src/unreviewed/payload.inc')
assert not allowed('src/unreviewed/payload.metalh')
assert content_findings(b'header\x00payload') == ['binary-source']
assert allowed('LICENSES/CuMetal-Apache-2.0.txt')
assert allowed('.github/CODEOWNERS')
assert allowed('.github/rulesets/main.json')
assert allowed('docs/REPOSITORY_GOVERNANCE.md')
assert allowed('docs/COMPLETE_SOURCE_BUILD.md')
assert allowed('docs/STATUS.md')
assert not allowed('.github/rulesets/unreviewed.json')
assert not allowed('LICENSES/weights.bin')
assert content_findings(('-----BEGIN ' + 'PRIVATE KEY-----').encode()) == ['private-key']
assert content_findings(b'not a PNG', image=True) == ['invalid-png']
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory) / 'repo'
    root.mkdir()
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    (root / 'README.md').write_text('Source preview\n')
    subprocess.run(['git', 'add', '.'], cwd=root, check=True)
    subprocess.run(['git', '-c', 'user.name=Export Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'fixture'], cwd=root, check=True)
    (root / 'work').mkdir()
    (root / 'work/weights.bin').write_bytes(b'model data')
    subprocess.run(['git', 'add', 'work'], cwd=root, check=True)
    output = Path(directory) / 'public'
    assert exporter.export(root, output) == (1, 1)
    assert not (output / '.git').exists() and not (output / 'work').exists()
    assert auditor.audit(output) == []
    (output / 'README.md').write_text('changed after export')
    assert any(x['reason'] == 'inventory-mismatch' for x in auditor.audit(output))
    (output / 'unexpected.dll').write_bytes(b'binary')
    assert any(x['reason'] == 'outside-source-policy' for x in auditor.audit(output))
    try:
        exporter.export(root, output)
        raise AssertionError('existing destination overwritten')
    except ValueError:
        pass
print('public-source tests passed')
