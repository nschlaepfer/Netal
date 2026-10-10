"""Pinned, model-independent scalar calibration data; not formula-derived constants."""
import hashlib
import json
from pathlib import Path

PROFILES = {
    'rcp': {'bytes': 2097152, 'sha256': '7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750',
            'nvidia': '370412e9639c02e57b71499cea5755bdc8606fa05fd6f4bdcd6051fe3ebe2005',
            'm3': '5e6b6283ff87e00a4688a5303910e3f29e6c2b54d56a3d9ca5dd1c332a2cf0d5',
            'exponents': [127,128], 'deltas': [0,-1,1]},
    'rsq': {'bytes': 4194304, 'sha256': 'd6792f533b996232cb49b02e82d2b16069e359ba3a0a735800685923a5c08a89',
            'nvidia': '716da5eb39165d9dd65fb94fc9f49448afa58234e7efb340edf78913fa2db109',
            'm3': '86a29b2748afa1d44e36eb5333f42a1cb1dd8ae7bbc9c5cb1ffea9207d884f3f',
            'exponents': [128,129], 'deltas': [0,-1,1,-2]},
}
BINARY_FILES = {f'runtime/arithmetic-calibration/{op}-correction-rom.bin': profile for op,profile in PROFILES.items()}
ORACLE_SOURCES = {'tools/math-calibration/oracle-rcp.cu', 'tools/math-calibration/oracle-rsq.cu', 'tools/math-calibration/lg2-characterization.cu'}


def validate_blob(path, data):
    profile = BINARY_FILES[path]
    if len(data) != profile['bytes'] or hashlib.sha256(data).hexdigest() != profile['sha256']:
        raise ValueError('Scalar calibration identity mismatch: '+path)


def load(root):
    root = Path(root)
    manifest = json.loads((root/'runtime/arithmetic-calibration/manifest.json').read_text())
    if manifest.get('schema') != 'netal.scalar-calibration.v1' or manifest.get('model_inputs') != 0 or manifest.get('profiles') != PROFILES:
        raise ValueError('Scalar calibration manifest drift')
    result = {}
    for relative in BINARY_FILES:
        data = (root/relative).read_bytes(); validate_blob(relative,data)
        result[Path(relative).name] = data
    return result
