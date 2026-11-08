#!/usr/bin/env python3
"""Exercise mobile Pro-chip admission without models or a physical device."""
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / 'src/ios_host'


def main():
    with tempfile.TemporaryDirectory(prefix='netal-ios-policy-') as temporary:
        binary = Path(temporary) / 'policy-test'
        subprocess.run([
            os.environ.get('CXX', 'clang++'), '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
            '-I', str(POLICY), str(POLICY / 'netal_ios_sr_device_policy.cpp'),
            str(POLICY / 'netal_ios_sr_device_policy_test.cpp'), '-o', str(binary),
        ], check=True)
        subprocess.run([str(binary)], check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
