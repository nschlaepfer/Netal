#!/usr/bin/env python3
"""Create a labelled external-color diagnostic, never a model/reference output.

Parameter-free per-channel Reinhard followed by sRGB. This is NOT an established
NVIDIA input contract or the game's tone mapper, and is not installed in runtime.
"""
import argparse
import hashlib
import json
import pathlib
import platform
import numpy as np


def transform(rgba):
    if rgba.shape[-1] != 4 or not np.isfinite(rgba).all():
        raise ValueError('finite RGBA required')
    if (rgba[..., :3] < 0).any():
        raise ValueError('negative RGB is outside this diagnostic domain')
    rgb = rgba[..., :3].astype(np.float64)
    linear = rgb/(1.0+rgb)
    encoded = np.where(linear <= 0.0031308, 12.92*linear,
                       1.055*np.power(linear, 1.0/2.4)-0.055)
    out = rgba.astype('<f2', copy=True)
    out[..., :3] = encoded.astype('<f2')
    if not np.isfinite(out).all() or (out[..., :3] < 0).any() or (out[..., :3] > 1).any():
        raise ValueError('bounded finite output invariant failed')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('external_rgba16f')
    ap.add_argument('new_output_directory')
    ap.add_argument('--input-sha256', required=True)
    args = ap.parse_args()
    source = pathlib.Path(args.external_rgba16f).resolve()
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if len(raw) != 1920*1080*8 or digest != args.input_sha256:
        raise ValueError('external input size/hash mismatch')
    original = np.frombuffer(raw, dtype='<f2').reshape(1080, 1920, 4)
    output = transform(original)
    if original.tobytes() != raw:
        raise ValueError('source modified')
    result = output.tobytes()
    out = pathlib.Path(args.new_output_directory).resolve()
    out.mkdir(parents=True, exist_ok=False)
    with (out/'input-color.rgba16le').open('xb') as f:
        f.write(result)
    receipt = {'schema': 'netal.nr.external-bounded-color-control.v1',
        'source': str(source), 'source_sha256': digest,
        'output_sha256': hashlib.sha256(result).hexdigest(),
        'bytes': len(result), 'width': 1920, 'height': 1080,
        'format': 'RGBA16F little-endian', 'alpha_preserved': True,
        'transform': 'per-channel y=x/(1+x), then piecewise sRGB; final FP16 store',
        'source_rgb_range': [float(original[..., :3].min()), float(original[..., :3].max())],
        'output_rgb_range': [float(output[..., :3].min()), float(output[..., :3].max())],
        'preparation_code_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        'python': platform.python_version(), 'numpy': np.__version__,
        'nvidia_color_contract_established': False, 'runtime_modified': False,
        'model_executed': False, 'expected_output_ingress': False,
        'note': 'Offline external-input control only. Not game tone mapping or a promoted fix.'}
    with (out/'input-receipt.json').open('x') as f:
        json.dump(receipt, f, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
