#!/usr/bin/env python3
"""Model-free host reference regression tests; no GPU/device result is implied."""
from pathlib import Path
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'src/ios_fp8_lab'
with tempfile.TemporaryDirectory(prefix='netal-fp8-reference-') as temporary:
    binary=Path(temporary)/'reference-test'
    subprocess.run(['clang++','-std=c++17','-Wall','-Wextra','-Werror',
                    str(SOURCE/'NetalFP8ReferenceTest.cpp'),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)
print('result=pass host_reference_only=true physical_device_executed=false')
