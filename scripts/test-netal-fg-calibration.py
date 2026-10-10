#!/usr/bin/env python3
"""Check the bounded FG calibration identities and source-export exception."""
import json
from pathlib import Path
import shutil
import tempfile
from netal_fg_calibration import BINARY_FILES, load, validate_blob
from public_source_policy import allowed, content_findings

ROOT = Path(__file__).resolve().parents[1]


def reject(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Invalid calibration accepted')


load(ROOT)
for relative in BINARY_FILES:
    data = (ROOT / relative).read_bytes()
    assert allowed(relative) and content_findings(data, relative=relative) == []
    for modified in (data[:-1], bytes([data[0] ^ 1]) + data[1:]):
        reject(lambda: validate_blob(relative, modified))
        assert content_findings(modified, relative=relative) == ['calibration-identity-mismatch']
assert not allowed('runtime/fg-arithmetic-calibration/extra.bin')
assert not allowed('runtime/fg-arithmetic-calibration/model.bin')
assert allowed('tools/math-calibration/fg-scalar-tables.cu')
assert not allowed('tools/math-calibration/unreviewed.cu')
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    destination = root / 'runtime/fg-arithmetic-calibration'
    shutil.copytree(ROOT / 'runtime/fg-arithmetic-calibration', destination)
    manifest = destination / 'manifest.json'
    original = manifest.read_bytes()
    value = json.loads(original)
    value['native_baselines']['unit']['sha256'] = '0' * 64
    manifest.write_text(json.dumps(value))
    reject(lambda: load(root))
    manifest.write_bytes(original)
    table = destination / 'sigmoid.bin'
    table.unlink()
    table.symlink_to(ROOT / 'runtime/fg-arithmetic-calibration/sigmoid.bin')
    reject(lambda: load(root))
print('FG calibration: full identities, corruption, baseline drift, symlinks and export boundaries passed')
