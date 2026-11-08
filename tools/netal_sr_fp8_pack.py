"""Private, reproducible E4M3FN derivatives of Netal's supported SR model.

Only active row-major GEMM/MLP matrices are packed. This module contains no model
data and does not validate inference, image quality, or hardware acceleration.
"""
from bisect import bisect_left
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'netal.sr.quantized-pack.v1'
MODEL_NAME = 'DLSS_SR_Feature1_Quality_K'
KEEP_FP16 = frozenset({'enc0.W1', 'dec0.W7'})
STAGES = ('enc0', 'enc1', 'enc2', 'enc3', 'enc4', 'dec5', 'dec4', 'dec3', 'dec2', 'dec1', 'dec0')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_object)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError('Invalid positive integer: ' + name)
    return value


def align(value, multiple):
    return (value + multiple - 1) // multiple * multiple


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def decode_e4m3fn(code):
    """Decode OFP8 E4M3 finite encoding, including signed zero and two NaNs."""
    if type(code) is not int or not 0 <= code <= 255:
        raise ValueError('FP8 code must be an unsigned byte')
    magnitude = code & 127
    if magnitude == 127:
        return math.nan
    exponent, fraction = magnitude >> 3, magnitude & 7
    value = math.ldexp(fraction, -9) if exponent == 0 else math.ldexp(1 + fraction / 8, exponent - 7)
    return math.copysign(value, -1 if code & 128 else 1)


POSITIVE_E4M3FN = tuple(decode_e4m3fn(code) for code in range(127))


def encode_e4m3fn(value):
    """RNE, ties-to-even, saturate-finite. Reject nonfinite calibration inputs."""
    if not math.isfinite(value):
        raise ValueError('Nonfinite FP8 input')
    sign = 128 if math.copysign(1.0, value) < 0 else 0
    magnitude = abs(value)
    if magnitude >= 448:
        return sign | 126
    upper = bisect_left(POSITIVE_E4M3FN, magnitude)
    if upper == 0:
        return sign
    lower = upper - 1
    dl, du = magnitude - POSITIVE_E4M3FN[lower], POSITIVE_E4M3FN[upper] - magnitude
    selected = lower if dl < du or (dl == du and lower % 2 == 0) else upper
    return sign | selected


def matrix_bytes(blob, offset, k, n, stride):
    """Read logical K,N half values while excluding source padding."""
    for name, value in [('K', k), ('N', n), ('row stride', stride)]:
        positive(value, name)
    if type(offset) is not int or offset < 0 or offset % 2 or stride < n:
        raise ValueError('Unsupported matrix offset or row stride')
    if offset + 2 * ((k - 1) * stride + n) > len(blob):
        raise ValueError('Matrix reads beyond the verified source')
    return b''.join(blob[offset + row * stride * 2:offset + (row * stride + n) * 2] for row in range(k))


def quantize_matrix(logical, k, n):
    """Return transposed N,K FP8 with aligned rows and canonical FP32 scales.

    Input is logical contiguous little-endian FP16 K,N. Scales are calculated
    then rounded to stored binary32 BEFORE quantization/error measurement.
    """
    positive(k, 'K'); positive(n, 'N')
    if len(logical) != k * n * 2:
        raise ValueError('FP16 matrix byte count does not match K,N')
    values = [item[0] for item in struct.iter_unpack('<e', logical)]
    if any(not math.isfinite(value) for value in values):
        raise ValueError('Nonfinite source matrix')
    padded_k, padded_n = align(k, 32), align(n, 32)
    row_stride = align(k, 128)
    output = bytearray(padded_n * row_stride)
    scales = [1.0] * padded_n
    for column in range(n):
        maximum = max(abs(values[row * n + column]) for row in range(k))
        scales[column] = f32(maximum / 448) if maximum else 1.0
        if not math.isfinite(scales[column]) or scales[column] <= 0:
            raise ValueError('Unrepresentable positive FP32 dequantization scale')
    squared_error = squared_source = 0.0
    max_error = 0.0
    saturated = underflowed = 0
    for row in range(k):
        for column in range(n):
            value, scale = values[row * n + column], scales[column]
            normalized = f32(value / scale)
            code = encode_e4m3fn(normalized)
            output[column * row_stride + row] = code
            reconstructed = decode_e4m3fn(code) * scale
            error = abs(value - reconstructed)
            squared_error += error * error
            squared_source += value * value
            max_error = max(max_error, error)
            saturated += abs(normalized) > 448
            underflowed += value != 0 and code & 127 == 0
    stats = {'elements': k * n, 'squared_error': squared_error, 'squared_source': squared_source,
             'rms_error': math.sqrt(squared_error / (k * n)),
             'relative_rms_error': math.sqrt(squared_error / squared_source) if squared_source else 0.0,
             'max_abs_error': max_error, 'saturated_inputs': saturated,
             'nonzero_to_zero': underflowed, 'nonfinite_inputs': 0}
    return bytes(output), b''.join(struct.pack('<f', scale) for scale in scales), {
        'shape': [padded_n, padded_k], 'order': 'N,K', 'row_stride_bytes': row_stride,
        'padded_k': padded_k, 'padded_n': padded_n, 'stats': stats}


def canonical_graphs(layouts):
    """Recreate phase-zero graphs from public layouts/generators, not input JSON."""
    geometry = load_module('netal_fp8_public_geometry', ROOT / 'scripts/netal_sr_layouts.py')
    files, _ = geometry.profile(1920, 1080)
    with tempfile.TemporaryDirectory(prefix='netal-fp8-graph-') as temporary:
        base, layout_dir = Path(temporary) / 'bundle', Path(temporary) / 'layouts'
        layout_dir.mkdir()
        for destination in (base, base / 'profiles/1920x1080'):
            for relative, data in files.items():
                path = destination / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        shutil.copyfile(base / 'pwin/enc0/arguments-template.bin', base / 'pwin/enc0-arguments-template.bin')
        for layer, layout in layouts.items():
            (layout_dir / f'layout-{layer}.json').write_text(json.dumps(layout))
        env = {key: value for key, value in os.environ.items() if not key.startswith('PWIN_')}
        env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
        result = subprocess.run([sys.executable, str(ROOT / 'tools/pwin_pipeline/build-profile-pipelines.py'),
                                 str(base), '--layout-dir', str(layout_dir)],
                                cwd=ROOT, env=env, capture_output=True, text=True)
        if result.returncode:
            raise ValueError('Public canonical graph generation failed')
        return {layer: read_json(base / f'pipeline/dltss_pwin_{layer}_layer.json') for layer in STAGES}


def active_matrices(root, layouts, blobs, canonical):
    """Follow phase-zero GEMM/MLP operand bindings, never matrix-looking masks."""
    records, graph = {}, {}
    gemms = mlps = 0
    for layer in STAGES:
        relative = f'pipeline/dltss_pwin_{layer}_layer.json'
        data = (root / relative).read_bytes()
        graph[relative] = sha(data)
        pipeline = json.loads(data, object_pairs_hook=unique_object)
        if pipeline != canonical[layer]:
            raise ValueError('Runtime graph differs from public canonical descriptor: ' + layer)
        files = pipeline.get('files')
        if files != [{'name': 'pweights', 'path': f'pipeline/{layer}-weights.bin'}]:
            raise ValueError('Unsupported canonical pipeline file bindings: ' + layer)
        for index, dispatch in enumerate(pipeline['dispatches']):
            function = dispatch['function']
            constants = dispatch.get('constants', {})
            buffers = dispatch.get('buffers', [])
            def operand(binding, k, n, stride, role):
                selected = [item for item in buffers if item.get('index') == binding]
                if len(selected) != 1 or selected[0].get('name') != 'pweights':
                    raise ValueError('Unsupported weight buffer binding')
                offset = selected[0].get('offset')
                matching = [(name, entry) for name, entry in layouts[layer].items()
                            if entry.get('dtype') == 'half' and entry.get('offset') == offset
                            and entry.get('k') == k and entry.get('n') == n]
                if len(matching) != 1 or stride != n:
                    raise ValueError('Matrix binding disagrees with canonical repack layout')
                name, entry = matching[0]
                identity = layer + '.' + name
                logical = matrix_bytes(blobs[layer], offset, k, n, stride)
                record = {'id': identity, 'layer': layer, 'name': name, 'shape': [k, n],
                          'source': {'path': f'pipeline/{layer}-weights.bin', 'offset': offset,
                                     'dtype': 'float16-le', 'row_stride_elements': stride,
                                     'logical_sha256': sha(logical)}, 'uses': []}
                previous = records.setdefault(identity, record)
                if previous['source'] != record['source'] or previous['shape'] != record['shape']:
                    raise ValueError('Ambiguous active matrix identity')
                previous['uses'].append({'pipeline': relative, 'dispatch': index, 'function': function,
                                         'buffer_index': binding, 'role': role,
                                         'store_cols': dispatch.get('bytes', [{}])[0].get('u32', [0] * 7)[6]
                                         if function == 'pwin_gemm3_tokens' else 0})
            if function == 'pwin_gemm3_tokens':
                k, n = constants['0'], constants['1']
                operand(1, k, n, constants.get('8', 0) or n, 'gemm_weight')
                gemms += 1
            elif function == 'pwin_mlp2_tokens':
                k, n, hidden = constants['0'], constants['1'], constants['14']
                operand(1, k, hidden, hidden, 'mlp_expansion')
                operand(3, hidden, n, constants.get('15', 0) or n, 'mlp_contraction')
                mlps += 1
            elif function.startswith(('pwin_gemm', 'pwin_mlp')):
                raise ValueError('Unsupported active matrix operator: ' + function)
    if gemms != 42 or mlps != 6 or len(records) != 54:
        raise ValueError('Unsupported active matrix graph')
    if sum(item['shape'][0] * item['shape'][1] for item in records.values()) != 1579008:
        raise ValueError('Unsupported active matrix element inventory')
    if not KEEP_FP16 <= records.keys():
        raise ValueError('Required sensitive FP16 matrices are missing')
    return sorted(records.values(), key=lambda item: (STAGES.index(item['layer']), item['source']['offset'])), graph


def validate_runtime(directory):
    root = Path(directory)
    if root.is_symlink() or not root.is_dir() or (root / 'manifest.json').is_symlink():
        raise ValueError('Use a real supported runtime directory')
    runtime = load_module('netal_fp8_runtime_manifest', ROOT / 'scripts/netal_sr_runtime_manifest.py')
    manifest = read_json(root / 'manifest.json')
    if manifest.get('model') != MODEL_NAME:
        raise ValueError('Unsupported source model identity')
    runtime.verify_engine(root, manifest)
    rows = {row['path']: row for row in manifest['files']}
    model = {}
    for name, (size, digest) in runtime.MODEL.items():
        data = (root / name).read_bytes()
        row = rows.get(name, {})
        if len(data) != size or sha(data) != digest or row.get('bytes') != size or row.get('sha256') != digest:
            raise ValueError('Unsupported original model bytes: ' + name)
        model[name] = {'bytes': size, 'sha256': digest}
    weights = (root / 'model/weights.bin').read_bytes()
    repacker = load_module('netal_fp8_source_repack', ROOT / 'tools/pwin_pipeline/repack.py')
    layouts, blobs, provenance = {}, {}, {}
    for layer in STAGES:
        spec_path = ROOT / f'tools/pwin_pipeline/spec-{layer}.json'
        spec_bytes = spec_path.read_bytes()
        expected_blob, expected_layout = repacker.repack(weights, json.loads(spec_bytes, object_pairs_hook=unique_object))
        blob_path, layout_path = f'pipeline/{layer}-weights.bin', f'pipeline/layouts/layout-{layer}.json'
        actual_blob, layout_bytes = (root / blob_path).read_bytes(), (root / layout_path).read_bytes()
        if actual_blob != expected_blob or json.loads(layout_bytes, object_pairs_hook=unique_object) != expected_layout:
            raise ValueError('Runtime repack differs from supported source model/spec: ' + layer)
        layouts[layer], blobs[layer] = expected_layout, actual_blob
        provenance[str(spec_path.relative_to(ROOT))] = sha(spec_bytes)
        provenance[blob_path] = sha(actual_blob)
        provenance[layout_path] = sha(layout_bytes)
    matrices, graph = active_matrices(root, layouts, blobs, canonical_graphs(layouts))
    provenance['tools/pwin_pipeline/repack.py'] = sha((ROOT / 'tools/pwin_pipeline/repack.py').read_bytes())
    for path in [ROOT / 'scripts/netal_sr_layouts.py', ROOT / 'tools/pwin_pipeline/build-profile-pipelines.py',
                 ROOT / 'tools/pwin_pipeline/mpp_backend.py', *sorted((ROOT / 'tools/pwin_pipeline').glob('make-*.py'))]:
        provenance[str(path.relative_to(ROOT))] = sha(path.read_bytes())
    return matrices, blobs, {'model': model, 'asset_root_sha256': manifest['asset_root_sha256'],
                            'runtime_manifest_sha256': sha((root / 'manifest.json').read_bytes()),
                            'layout_and_repack_sha256': provenance, 'graph_sha256': graph}


def append_payload(payloads, path, data):
    blob = payloads.setdefault(path, bytearray())
    blob.extend(bytes(-len(blob) % 128))
    offset = len(blob)
    blob.extend(data)
    return {'file': path, 'offset': offset, 'bytes': len(data), 'sha256': sha(data), 'alignment': 128}


def make_pack(matrices, blobs, source):
    payloads, tensors = {}, []
    for item in matrices:
        entry = json.loads(json.dumps(item))
        k, n = entry['shape']
        logical = matrix_bytes(blobs[entry['layer']], entry['source']['offset'], k, n,
                               entry['source']['row_stride_elements'])
        if sha(logical) != entry['source']['logical_sha256']:
            raise ValueError('Source matrix changed during packing')
        prefix = 'pipeline/fp8/' + entry['layer']
        if entry['id'] in KEEP_FP16:
            entry['precision'] = 'float16-le'
            entry['reason'] = 'sensitive input/final projection retained in FP16; quality unvalidated'
            entry['payload'] = {**append_payload(payloads, prefix + '-kept.f16le.bin', logical),
                                'shape': [k, n], 'order': 'K,N', 'row_stride_bytes': n * 2}
            entry['scales'] = None
        else:
            packed, scales, details = quantize_matrix(logical, k, n)
            entry['precision'] = 'e4m3fn'
            entry['payload'] = {**append_payload(payloads, prefix + '-weights.e4m3fn.bin', packed),
                                **{key: value for key, value in details.items() if key != 'stats'}}
            entry['scales'] = {**append_payload(payloads, prefix + '-scales.f32le.bin', scales),
                               'count': details['padded_n'], 'dtype': 'float32-le',
                               'axis': 'output_column', 'group_size': 'entire_K',
                               'direction': 'decoded_value * dequant_scale', 'padding_value': 1.0}
            entry['weight_reconstruction'] = details['stats']
        tensors.append(entry)
    files = [{'path': path, 'bytes': len(data), 'sha256': sha(data)} for path, data in sorted(payloads.items())]
    quantized = [entry for entry in tensors if entry['precision'] == 'e4m3fn']
    squared_error = sum(entry['weight_reconstruction']['squared_error'] for entry in quantized)
    squared_source = sum(entry['weight_reconstruction']['squared_source'] for entry in quantized)
    manifest = {
        'schema': SCHEMA, 'source_model': MODEL_NAME, 'derived_approximate': True, 'dlss45': False,
        'source': source,
        'recipe': {'version': 1, 'format': 'OFP8 E4M3FN', 'rounding': 'round-to-nearest ties-to-even',
                   'overflow': 'saturate finite at +/-448', 'subnormals': 'preserved',
                   'nonfinite_inputs': 'rejected', 'negative_zero': 'preserved',
                   'scale': 'FP32(amax(abs(W[:,n]))/448); all-zero column uses 1',
                   'quantize': 'q=E4M3FN_RNE(source/dequant_scale)',
                   'dequantize': 'source_hat=decode(q)*dequant_scale',
                   'scale_rounding': 'stored float32 scale is used for quantization and reported error',
                   'quantizer_arithmetic': 'FP32 quotient followed by E4M3FN RNE; binary64 error statistics',
                   'padding': 'N and logical K rounded to32; physical K byte stride rounded to128; zero values/scale1',
                   'sensitive_fp16': sorted(KEEP_FP16),
                   'quantizer_source_sha256': sha(Path(__file__).read_bytes()),
                   'activation_quantization': 'not implemented/calibrated by this weight pack',
                   'split_handoffs': 'not changed by packing; a consumer must explicitly validate or declare relaxation'},
        'tensors': tensors, 'files': files,
        'summary': {'active_matrices': len(tensors), 'quantized_matrices': len(quantized),
                    'kept_fp16_matrices': len(tensors) - len(quantized),
                    'active_source_fp16_bytes': sum(entry['shape'][0] * entry['shape'][1] * 2 for entry in tensors),
                    'payload_bytes_including_padding': sum(len(data) for data in payloads.values()),
                    'quantized_weight_relative_rms_error': math.sqrt(squared_error / squared_source) if squared_source else 0.0},
        'claims': {'inference_validated': False, 'image_quality_validated': False,
                   'hardware_acceleration_validated': False, 'gameplay_validated': False,
                   'redistribution_cleared': False},
    }
    manifest['pack_root_sha256'] = sha(canonical_json(manifest))
    return {path: bytes(data) for path, data in payloads.items()}, manifest


def verify_payload_inventory(directory):
    """Validate inventory and tensor semantics only; this does NOT admit a model.

    Production consumers must call verify_pack with the original source runtime
    so self-rehashed payloads cannot masquerade as canonical model derivatives.
    """
    root = Path(directory)
    if root.is_symlink() or not root.is_dir() or (root / 'manifest.json').is_symlink():
        raise ValueError('Use a real quantized pack directory')
    manifest = read_json(root / 'manifest.json')
    if manifest.get('schema') != SCHEMA or manifest.get('source_model') != MODEL_NAME:
        raise ValueError('Unsupported quantized pack identity')
    if manifest.get('derived_approximate') is not True or manifest.get('dlss45') is not False:
        raise ValueError('Unsupported quantized pack claims')
    claims = manifest.get('claims')
    if not isinstance(claims, dict) or not claims or any(value is not False for value in claims.values()):
        raise ValueError('Weight packing does not establish runtime/quality/redistribution claims')
    supplied_root = manifest.get('pack_root_sha256')
    material = {key: value for key, value in manifest.items() if key != 'pack_root_sha256'}
    if supplied_root != sha(canonical_json(material)):
        raise ValueError('Quantized pack manifest root mismatch')
    files = manifest.get('files')
    if not isinstance(files, list):
        raise ValueError('Missing quantized pack file inventory')
    expected = {}
    for row in files:
        if not isinstance(row, dict):
            raise ValueError('Invalid quantized payload inventory row')
        path = row.get('path')
        if (not isinstance(path, str) or not path.startswith('pipeline/fp8/') or
                not all(character.isascii() and (character.isalnum() or character in '_./-') for character in path) or
                any(part in ('', '.', '..') for part in path.split('/')) or path in expected):
            raise ValueError('Unsafe or duplicate quantized payload path')
        expected[path] = row
    actual = set()
    payloads = {}
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('Linked quantized pack entry')
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == 'manifest.json':
            continue
        data = path.read_bytes(); actual.add(relative)
        row = expected.get(relative, {})
        if type(row.get('bytes')) is not int or row.get('bytes') != len(data) or row.get('sha256') != sha(data):
            raise ValueError('Quantized payload differs from inventory')
        payloads[relative] = data
    if actual != expected.keys():
        raise ValueError('Missing quantized payload')
    tensors = manifest.get('tensors')
    if not isinstance(tensors, list) or not tensors:
        raise ValueError('Missing active matrix inventory')
    seen, used_ranges = set(), {}
    for tensor in tensors:
        if not isinstance(tensor, dict) or not isinstance(tensor.get('id'), str) or tensor['id'] in seen:
            raise ValueError('Duplicate quantized tensor')
        seen.add(tensor.get('id'))
        shape = tensor.get('shape')
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValueError('Invalid logical matrix shape')
        k, n = (positive(value, 'logical shape') for value in shape)
        precision = tensor.get('precision')
        if precision not in ('e4m3fn', 'float16-le') or (precision == 'float16-le') != (tensor['id'] in KEEP_FP16):
            raise ValueError('Unsupported sensitive-matrix precision selection')
        slices = {}
        for key in ('payload', 'scales'):
            entry = tensor.get(key)
            if entry is None and key == 'scales' and tensor.get('precision') == 'float16-le':
                continue
            if not isinstance(entry, dict) or entry.get('file') not in payloads:
                raise ValueError('Missing tensor payload metadata')
            offset, size = entry.get('offset'), entry.get('bytes')
            if type(offset) is not int or type(size) is not int or offset < 0 or size <= 0 or offset % 128:
                raise ValueError('Invalid tensor payload bounds/alignment')
            data = payloads[entry['file']]
            if offset + size > len(data) or sha(data[offset:offset + size]) != entry.get('sha256'):
                raise ValueError('Tensor payload differs from metadata')
            ranges = used_ranges.setdefault(entry['file'], [])
            if any(offset < end and offset + size > begin for begin, end in ranges):
                raise ValueError('Overlapping tensor payload ranges')
            ranges.append((offset, offset + size))
            slices[key] = data[offset:offset + size]
        payload = tensor['payload']
        if precision == 'float16-le':
            if payload.get('shape') != [k, n] or payload.get('order') != 'K,N' or payload.get('row_stride_bytes') != n * 2 or len(slices['payload']) != k * n * 2:
                raise ValueError('Unsupported retained FP16 layout')
            if any(not math.isfinite(value) for value, in struct.iter_unpack('<e', slices['payload'])):
                raise ValueError('Nonfinite retained FP16 matrix')
        else:
            pn, pk, stride = align(n, 32), align(k, 32), align(k, 128)
            if (payload.get('shape') != [pn, pk] or payload.get('order') != 'N,K' or
                    payload.get('padded_k') != pk or payload.get('padded_n') != pn or
                    payload.get('row_stride_bytes') != stride or len(slices['payload']) != pn * stride):
                raise ValueError('Unsupported transposed FP8 layout')
            scales = tensor['scales']
            if (scales.get('count') != pn or scales.get('dtype') != 'float32-le' or
                    scales.get('axis') != 'output_column' or scales.get('group_size') != 'entire_K' or
                    scales.get('direction') != 'decoded_value * dequant_scale' or
                    scales.get('padding_value') != 1 or len(slices['scales']) != pn * 4):
                raise ValueError('Unsupported FP32 scale contract')
            decoded_scales = struct.unpack('<' + str(pn) + 'f', slices['scales'])
            if any(not math.isfinite(value) or value <= 0 for value in decoded_scales) or any(value != 1 for value in decoded_scales[n:]):
                raise ValueError('FP32 dequantization scales must be finite/positive with unit padding')
            packed = slices['payload']
            for column in range(pn):
                row = packed[column * stride:(column + 1) * stride]
                if any(code & 127 == 127 for code in row[:k] if column < n):
                    raise ValueError('Nonfinite E4M3FN code')
                if any(row[k:] if column < n else row):
                    raise ValueError('Nonzero FP8 padding')
    for path, data in payloads.items():
        consumed = bytearray(len(data))
        for begin, end in used_ranges.get(path, []):
            consumed[begin:end] = b'\1' * (end - begin)
        if any(value for value, covered in zip(data, consumed) if not covered):
            raise ValueError('Nonzero inter-tensor alignment padding')
    if len(seen) != manifest.get('summary', {}).get('active_matrices'):
        raise ValueError('Quantized tensor count mismatch')
    return manifest


def verify_pack(directory, runtime_directory):
    """Admit only the deterministic derivative of the exact supported runtime."""
    manifest = verify_payload_inventory(directory)
    summary = manifest.get('summary', {})
    if (summary.get('active_matrices'), summary.get('quantized_matrices'), summary.get('kept_fp16_matrices')) != (54, 52, 2):
        raise ValueError('Unsupported active matrix counts')
    matrices, blobs, source = validate_runtime(runtime_directory)
    payloads, expected = make_pack(matrices, blobs, source)
    if manifest != expected:
        raise ValueError('Pack is not the canonical quantized derivative of the supplied supported runtime')
    for relative, data in payloads.items():
        if (Path(directory) / relative).read_bytes() != data:
            raise ValueError('Pack payload is not derived from the supplied supported runtime')
    return manifest


def build(runtime_directory, output_directory):
    runtime, output = Path(runtime_directory), Path(output_directory).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError('Output exists; choose a new private directory')
    if output.resolve().is_relative_to(runtime.resolve()):
        raise ValueError('Output must be outside the source runtime')
    matrices, blobs, source = validate_runtime(runtime)
    payloads, manifest = make_pack(matrices, blobs, source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.netal-sr-fp8-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'pack'; stage.mkdir(mode=0o700)
        for relative, data in payloads.items():
            path = stage / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        verify_payload_inventory(stage)
        if output.exists() or output.is_symlink():
            raise ValueError('Output appeared during packing; preserved')
        stage.rename(output)
    return manifest
