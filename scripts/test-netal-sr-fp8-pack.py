#!/usr/bin/env python3
"""Model-free codec, transposition, scale, integrity and rejection checks.

Synthetic pack fixtures exercise integrity only. They are not accepted by the
supported-runtime builder and do not constitute real-model or inference proof.
"""
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import netal_sr_fp8_pack as fp8


def halves(values):
    return b''.join(struct.pack('<e', value) for value in values)


class CodecTests(unittest.TestCase):
    def test_every_code_and_special(self):
        for code in range(256):
            value = fp8.decode_e4m3fn(code)
            if code & 127 == 127:
                self.assertTrue(math.isnan(value))
            else:
                self.assertEqual(fp8.encode_e4m3fn(value), code)
        self.assertEqual(fp8.decode_e4m3fn(126), 448)
        self.assertEqual(fp8.decode_e4m3fn(1), 2 ** -9)
        self.assertEqual(fp8.encode_e4m3fn(100000), 126)
        self.assertEqual(fp8.encode_e4m3fn(-100000), 254)
        for value in [math.nan, math.inf, -math.inf]:
            with self.assertRaises(ValueError):
                fp8.encode_e4m3fn(value)
        for code in [-1, 256, True, 1.0]:
            with self.assertRaises(ValueError):
                fp8.decode_e4m3fn(code)

    def test_all_adjacent_ties_and_neighbors(self):
        for lower in range(126):
            upper = lower + 1
            midpoint = (fp8.decode_e4m3fn(lower) + fp8.decode_e4m3fn(upper)) / 2
            even = lower if lower % 2 == 0 else upper
            self.assertEqual(fp8.encode_e4m3fn(midpoint), even)
            self.assertEqual(fp8.encode_e4m3fn(math.nextafter(midpoint, 0)), lower)
            self.assertEqual(fp8.encode_e4m3fn(math.nextafter(midpoint, math.inf)), upper)
            self.assertEqual(fp8.encode_e4m3fn(-midpoint), even | 128)

    def test_exhaustive_finite_fp16_against_integer_distance_oracle(self):
        # Half values and E4M3FN are exact integers in units of 2**-24. This
        # brute-force integer nearest-distance oracle does not share bisect or
        # floating-distance rounding with the production codec.
        units = [int(fp8.decode_e4m3fn(code) * 2 ** 24) for code in range(127)]
        for bits in range(0x7c00):
            exponent, fraction = bits >> 10, bits & 1023
            integer = fraction if exponent == 0 else (1024 + fraction) << (exponent - 1)
            expected = min(range(127), key=lambda code: (abs(units[code] - integer), code & 1))
            value = struct.unpack('<e', struct.pack('<H', bits))[0]
            self.assertEqual(fp8.encode_e4m3fn(value), expected, hex(bits))
            self.assertEqual(fp8.encode_e4m3fn(-value), expected | 128, hex(bits | 0x8000))


class PackingTests(unittest.TestCase):
    def test_strided_source_and_bounds(self):
        source = halves([1, 2, 99, 3, 4])
        self.assertEqual(fp8.matrix_bytes(source, 0, 2, 2, 3), halves([1, 2, 3, 4]))
        for args in [(0, 0, 2, 2), (0, 2, 2, 1), (1, 2, 2, 3),
                     (-2, 2, 2, 3), (4, 2, 2, 3), (0, True, 2, 2)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                fp8.matrix_bytes(source, *args)

    def test_column_scales_transposition_and_padding(self):
        values = [-1, 0, 2, 0.5, -0.0, 4, 1, 0, -8]
        payload, raw_scales, info = fp8.quantize_matrix(halves(values), 3, 3)
        scales = struct.unpack('<32f', raw_scales)
        self.assertEqual(info['shape'], [32, 32])
        self.assertEqual(info['row_stride_bytes'], 128)
        self.assertEqual(len(payload), 32 * 128)
        self.assertEqual(scales[1], 1)
        self.assertEqual(scales[3:], (1.0,) * 29)
        errors = []
        for row in range(3):
            for column in range(3):
                code = payload[column * 128 + row]
                self.assertEqual(code, fp8.encode_e4m3fn(fp8.f32(values[row * 3 + column] / scales[column])))
                errors.append(abs(values[row * 3 + column] - fp8.decode_e4m3fn(code) * scales[column]))
        for column in range(32):
            start = column * 128 + (3 if column < 3 else 0)
            self.assertFalse(any(payload[start:(column + 1) * 128]))
        self.assertAlmostEqual(info['stats']['max_abs_error'], max(errors))
        self.assertEqual(info['stats']['nonfinite_inputs'], 0)
        for value in [math.inf, math.nan]:
            with self.assertRaises(ValueError):
                fp8.quantize_matrix(halves([value]), 1, 1)
        with self.assertRaises(ValueError):
            fp8.quantize_matrix(b'', 1, 1)

    def synthetic_pack(self, root):
        raw = halves([1, 2, 3, 4])
        matrices = [{'id': 'synthetic.W', 'layer': 'synthetic', 'name': 'W', 'shape': [2, 2],
                     'source': {'path': 'pipeline/synthetic-weights.bin', 'offset': 0,
                                'dtype': 'float16-le', 'row_stride_elements': 2,
                                'logical_sha256': fp8.sha(raw)}, 'uses': []}]
        payloads, manifest = fp8.make_pack(matrices, {'synthetic': raw}, {'fixture': 'model-free integrity test'})
        for name, data in payloads.items():
            path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        (root / 'manifest.json').write_text(json.dumps(manifest))
        return payloads, manifest

    def test_deterministic_metadata_and_file_integrity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads, manifest = self.synthetic_pack(root)
            verified = fp8.verify_payload_inventory(root)
            self.assertEqual(verified['pack_root_sha256'], manifest['pack_root_sha256'])
            self.assertTrue(verified['derived_approximate'])
            self.assertFalse(verified['claims']['inference_validated'])
            path = root / next(iter(payloads))
            original = path.read_bytes(); path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
            with self.assertRaises(ValueError):
                fp8.verify_payload_inventory(root)
            path.write_bytes(original); path.unlink()
            with self.assertRaises(ValueError):
                fp8.verify_payload_inventory(root)

    def test_manifest_rejections(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, manifest = self.synthetic_pack(root)
            manifest['source_model'] = 'unrecognized-model'
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                fp8.verify_payload_inventory(root)
            (root / 'manifest.json').write_text('{"schema":"one","schema":"two"}')
            with self.assertRaises(ValueError):
                fp8.verify_payload_inventory(root)

    def test_self_rehashed_semantic_corruption_is_rejected(self):
        def reseal(root, manifest):
            for row in manifest['files']:
                raw = (root / row['path']).read_bytes()
                row['bytes'], row['sha256'] = len(raw), fp8.sha(raw)
            for tensor in manifest['tensors']:
                for key in ('payload', 'scales'):
                    part = tensor.get(key)
                    if part:
                        raw = (root / part['file']).read_bytes()
                        part['sha256'] = fp8.sha(raw[part['offset']:part['offset'] + part['bytes']])
            manifest.pop('pack_root_sha256', None)
            manifest['pack_root_sha256'] = fp8.sha(fp8.canonical_json(manifest))
            (root / 'manifest.json').write_text(json.dumps(manifest))
        for corruption in ['zero-scale', 'negative-scale', 'nan-scale', 'nan-code', 'padding',
                           'shape', 'stride', 'scale-count', 'precision', 'claim', 'empty']:
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _, manifest = self.synthetic_pack(root)
                tensor = manifest['tensors'][0]
                if corruption.endswith('scale'):
                    part = tensor['scales']; path = root / part['file']; raw = bytearray(path.read_bytes())
                    value = {'zero-scale': 0, 'negative-scale': -1, 'nan-scale': math.nan}[corruption]
                    struct.pack_into('<f', raw, part['offset'], value); path.write_bytes(raw)
                elif corruption in ('nan-code', 'padding'):
                    part = tensor['payload']; path = root / part['file']; raw = bytearray(path.read_bytes())
                    raw[part['offset'] + (0 if corruption == 'nan-code' else 2)] = 127 if corruption == 'nan-code' else 1
                    path.write_bytes(raw)
                elif corruption == 'shape': tensor['shape'] = [2, 33]
                elif corruption == 'stride': tensor['payload']['row_stride_bytes'] = 256
                elif corruption == 'scale-count': tensor['scales']['count'] = 1
                elif corruption == 'precision': tensor['precision'] = 'float16-le'
                elif corruption == 'claim': manifest['claims']['inference_validated'] = True
                elif corruption == 'empty':
                    manifest['files'], manifest['tensors'] = [], []
                    manifest['summary']['active_matrices'] = 0
                    for path in root.rglob('*.bin'): path.unlink()
                reseal(root, manifest)
                with self.assertRaises(ValueError):
                    fp8.verify_payload_inventory(root)

    def test_inventory_integrity_does_not_admit_synthetic_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.synthetic_pack(root)
            fp8.verify_payload_inventory(root)
            with self.assertRaises(ValueError):
                fp8.verify_pack(root, root / 'nonexistent-source-runtime')

    def test_source_recomputation_rejects_self_consistent_wrong_derivative(self):
        # Mock only source acquisition: test the real production recomputation
        # and rejection logic without distributing the proprietary source model.
        raw = halves([1, 2, 3, 4])
        matrices, blobs = [], {}
        for identity in ['enc0.W1', 'dec0.W7'] + [f'synthetic.W{index}' for index in range(52)]:
            layer, name = identity.split('.')
            blobs[layer] = raw
            matrices.append({'id': identity, 'layer': layer, 'name': name, 'shape': [2, 2],
                             'source': {'path': f'pipeline/{layer}-weights.bin', 'offset': 0,
                                        'dtype': 'float16-le', 'row_stride_elements': 2,
                                        'logical_sha256': fp8.sha(raw)}, 'uses': []})
        source = {'fixture': 'synthetic deterministic derivation, not supported-model validation'}
        payloads, manifest = fp8.make_pack(matrices, blobs, source)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, data in payloads.items():
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with mock.patch.object(fp8, 'validate_runtime', return_value=(matrices, blobs, source)):
                fp8.verify_pack(root, root / 'synthetic-runtime')
                target = next(tensor for tensor in manifest['tensors'] if tensor['precision'] == 'e4m3fn')
                scale = target['scales']; path = root / scale['file']; data = bytearray(path.read_bytes())
                struct.pack_into('<f', data, scale['offset'], 2.0); path.write_bytes(data)
                for row in manifest['files']:
                    if row['path'] == scale['file']: row['sha256'] = fp8.sha(data)
                scale['sha256'] = fp8.sha(data[scale['offset']:scale['offset'] + scale['bytes']])
                manifest.pop('pack_root_sha256')
                manifest['pack_root_sha256'] = fp8.sha(fp8.canonical_json(manifest))
                (root / 'manifest.json').write_text(json.dumps(manifest))
                fp8.verify_payload_inventory(root)  # Self-consistent, finite and structurally valid.
                with self.assertRaisesRegex(ValueError, 'canonical quantized derivative'):
                    fp8.verify_pack(root, root / 'synthetic-runtime')

    def test_supported_runtime_builder_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); runtime = root / 'input'; runtime.mkdir()
            (runtime / 'manifest.json').write_text(json.dumps({'schema': 'netal.sr.runtime.v2', 'model': 'unsupported'}))
            with self.assertRaises(ValueError):
                fp8.build(runtime, root / 'output')
            self.assertFalse((root / 'output').exists())
            with self.assertRaises(ValueError):
                fp8.build(runtime, runtime / 'nested')
            (root / 'existing').mkdir()
            with self.assertRaises(ValueError):
                fp8.build(runtime, root / 'existing')


if __name__ == '__main__':
    unittest.main()
