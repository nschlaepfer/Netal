"""Pinned standalone FG arithmetic data, separate from model buffers."""
import hashlib
import json
from pathlib import Path

PROFILES = {
    'ex2-unit.bin': {'bytes': 4194305, 'sha256': '60c2ae1a0869ce6f10ba093a33def68ecdbf4ba0d789cd186289586b63b9f234'},
    'ex2-fraction.bin': {'bytes': 2097153, 'sha256': '76bacb1adf8e53accdd6a40a64940c073f001e091232e9119b155f2ab25f1a13'},
    'sigmoid.bin': {'bytes': 262144, 'sha256': '394394a5258bad437495d68076be75d8413fa0ed5b752400e3947c332437b850'},
}
BASELINES = {
    'unit': {'records': 16777217, 'bytes': 67108868, 'sha256': '41e853d34dfea8d2fc3a75e1ab8c2790f526bed3c1f1ccd1b0df3be3eb22a381'},
    'fraction': {'records': 8388609, 'bytes': 33554436, 'sha256': 'b8f55ae73000851d89fd0bcc14e4178f353598bb48f2634cd919794e5258e092'},
}
BINARY_FILES = {'runtime/fg-arithmetic-calibration/' + name: profile for name, profile in PROFILES.items()}
ORACLE_SOURCES = {'tools/math-calibration/fg-scalar-tables.cu'}


def validate_blob(relative, data):
    expected = BINARY_FILES[relative]
    if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
        raise ValueError('FG scalar calibration identity mismatch: ' + relative)


def load(root):
    root = Path(root)
    manifest = json.loads((root / 'runtime/fg-arithmetic-calibration/manifest.json').read_text())
    if (manifest.get('schema') != 'netal.fg.scalar-calibration.v1'
            or manifest.get('model_inputs') != 0 or manifest.get('profiles') != PROFILES
            or manifest.get('native_baselines') != BASELINES):
        raise ValueError('FG scalar calibration manifest drift')
    for relative in BINARY_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError('Missing or linked FG scalar calibration: ' + relative)
        validate_blob(relative, path.read_bytes())
    return manifest
