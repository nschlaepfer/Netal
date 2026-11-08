#!/usr/bin/env python3
"""Build original public PWin kernels plus the opt-in approximate MPP GEMM.

Compilation is not inference validation, a device benchmark, or confirmation of
neural-accelerator execution. No model, weights, or translated reference body is
included. Pair with build-profile-pipelines.py --gemm-backend mpp.
"""
import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path
from netal_apple_build import add_metal_target_arguments, metal_toolchain

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("pwin_gemm.metal pwin_gemm2.metal pwin_gemm3.metal pwin_norm.metal "
           "pwin_mix.metal pwin_mix2.metal pwin_mix3.metal pwin_mix4.metal pwin_mix5.metal "
           "pwin_mixp.metal pwin_mixp2.metal pwin_mixm.metal pwin_mixm2.metal pwin_merge.metal "
           "pwin_d2s.metal pwin_pad.metal pwin_pad2.metal pwin_mlp.metal pwin_mlp2.metal "
           "pwin_attn.metal pwin_dbg.metal pwin_gemm_mpp.metal").split()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', '--output', dest='output', type=Path, required=True)
    parser.add_argument('--variant', choices=('staged', 'direct'), default='staged',
                        help='staged preserves the baseline; direct uses experimental device-tensor MAC')
    add_metal_target_arguments(parser)
    args = parser.parse_args()
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        parser.error('Output exists; choose a new directory')
    try:
        compiler, linker, target_flags, target_receipt = metal_toolchain(args.sdk, args.deployment_target)
    except ValueError as error:
        parser.error(str(error))
    records, entries, texts = [], [], []
    for name in SOURCES:
        path = ROOT / 'tools/pwin_pipeline' / name
        source = path.read_text()
        entries.extend(re.findall(r'kernel\s+void\s+(\w+)\s*\(', source))
        records.append({'source': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        texts.append('// source: ' + name + '\n' + source + '\n')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.sr-mpp-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'output'
        stage.mkdir()
        source = Path(temporary) / 'pwin_mpp_all.metal'
        source.write_text(''.join(texts))
        air = Path(temporary) / 'pwin_mpp_all.air'
        library = stage / 'pwin_mpp_all.metallib'
        flags = ['-std=metal4.0', '-fno-fast-math', '-DPWIN_MPP_SHARED_SHAPES=1']
        if args.variant == 'direct': flags.append('-DPWIN_MPP_DIRECT=1')
        subprocess.run([str(compiler), *target_flags, *flags, '-c', str(source), '-o', str(air)], check=True)
        subprocess.run([str(linker), str(air), '-o', str(library)], check=True)
        receipt = {'schema': 'netal.sr.mpp-kernels.v1', **target_receipt,
                   'compiler': subprocess.check_output([str(compiler), '--version'], text=True).strip(),
                   'flags': flags, 'sources': records, 'kernel_entries': sorted(set(entries)),
                   'library': library.name, 'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
                   'gemm_backend': 'mpp', 'mpp_variant': args.variant, 'candidate_approximate': True,
                   'reference_exact': False, 'model_family': 'DLSS4 K', 'dlss45': False,
                   'non_gemm_operators': 'original public PWin kernels',
                   'gemm_tile': {'tokens': 64, 'columns': 32, 'k': 16, 'threads': 128, 'threadgroup_bytes': 3072 if args.variant == 'staged' else 0},
                   'complete_runtime': False, 'inference_validated': False,
                   'physical_device_validated': False, 'acceleration_verified': False}
        (stage / 'build-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        stage.rename(output)
    print(f'result=pass library={library.name} kernels={len(set(entries))}')


if __name__ == '__main__':
    main()
