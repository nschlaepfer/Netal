#!/usr/bin/env python3
"""Compare an encoder prefix without promoting a single smoke run to acceptance."""
import argparse
import hashlib
import json
from pathlib import Path

RUNTIME = 'e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e'
GRAPH = '3de475044e840d59f0235f0ff12763e99950909f81e5ae8b8c832375a25c88c5'


def sha(data): return hashlib.sha256(data).hexdigest()
def file_sha(path): return sha(path.read_bytes())
def require(condition, message):
    if not condition: raise ValueError(message)
def checked_receipt(path):
    r = json.loads(path.read_text())
    body = dict(r); root = body.pop('receipt_root_sha256')
    require(sha(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()) == root, f'receipt drift: {path}')
    return r


def pack_native_reference(canonical, width, height, channels, planar=False):
    """Invert the diagnostic layout; caller also pins the original raw RTX hash."""
    raw = bytearray(width * height * channels)
    for y in range(height):
        for x in range(width):
            row = ((y // 8) * ((width + 7) // 8) + x // 8) * 64 + ((x % 8) // 4) * 32 + ((y % 8) // 4) * 16 + (y % 4) * 4 + x % 4
            if planar:
                for c in range(0, channels, 16):
                    dst = ((c // 16) * width * height + y * width + x) * 16
                    src = row * channels + c
                    raw[dst:dst+16] = canonical[src:src+16]
            else:
                cell = (y // 4) * (width // 4) + x // 4; p = (y % 4) * 4 + x % 4
                for c in range(0, channels, 4):
                    dst = cell * 16 * channels + (c // 32) * 512 + (p % 8) * 64 + ((c % 16) // 4) * 16 + (p // 8) * 4 + ((c % 32) // 16) * 8
                    src = row * channels + c
                    raw[dst:dst+4] = canonical[src:src+4]
    return bytes(raw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--attempt', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--single-run', help='Explicit non-accepted diagnostic, e.g. smoke. No repeat claims.')
    p.add_argument('--last-block', type=int, choices=(8, 14, 22), default=8)
    a = p.parse_args()
    require(not a.output.exists(), 'preserve previous verification')
    repo = Path(__file__).resolve().parents[1]
    reference = checked_receipt(a.reference / 'receipt.json')
    require(reference['runtime_sha256'] == RUNTIME and reference['graph_root_sha256'] == GRAPH, 'model identity')
    require(reference['comparison_only'] is True and reference['full_model_inference'] is False, 'reference scope')
    require([b['block'] for b in reference['blocks']] == list(range(1, a.last_block + 1)), 'reference prefix scope')
    targets = {}; weight_hashes = {}
    for block in reference['blocks']:
        b = block['block']; weight_hashes[b] = block['weight_sha256']
        require(file_sha(a.reference / f'weights/block{b}.bin') == weight_hashes[b], 'weight drift')
        for name, digest in block['outputs'].items():
            data = (a.reference / name).read_bytes()
            require(sha(data) == digest, 'reference tensor drift: ' + name)
            targets[name] = data
    expected_total = {8:119439360, 14:148193280, 22:167198720}[a.last_block]
    require(len(targets) == {8:10, 14:17, 22:26}[a.last_block] and sum(map(len, targets.values())) == expected_total, 'reference output coverage')
    native_targets = {}
    if a.last_block == 22:
        for block in reference['blocks']:
            b = block['block']
            if b in (4,8,14,22):
                data = pack_native_reference(targets[f'block{b}.e4m3'], block['width'], block['height'], block['channels'])
                require(sha(data) == block['raw_output_sha256'], 'original packed RTX hash not reconstructed')
                native_targets[f'block{b}-native.e4m3'] = data
            if b == 22:
                data = pack_native_reference(targets['block22-downsample.e4m3'], 60, 36, 512, True)
                require(sha(data) == block['downsample_raw_sha256'], 'original middle input hash not reconstructed')
                native_targets['block22-middle.e4m3'] = data
        require(sum(map(len,native_targets.values())) == 34283520, 'native interface coverage')
    entry_ref = checked_receipt(repo / 'build/nr_swin_block/reference-v2/receipt.json')
    input_sha = file_sha(repo / 'build/nr_swin_block/reference-v2/input.e4m3')
    require(input_sha == entry_ref['flat_input_sha256'], 'recorded Block0 input drift')
    names = [a.single_run] if a.single_run else ['a5-a', 'a5-b', '5a-a', '5a-b']
    pids = set(); poisons = []; runs = []; samples = {}; total = 0; native_total = 0
    for name in names:
        run = a.attempt / name
        r = json.loads((run / 'run.json').read_text())
        require(r['schema'] == f'neuralstudio.nr.prefix{a.last_block}-live-chain.v1' and r['result'] == 'pass', 'incomplete/wrong run')
        require(r['pid'] not in pids, 'reused native process'); pids.add(r['pid'])
        require(r['poison'] in (0xa5, 0x5a), 'unsupported poison'); poisons.append(r['poison'])
        if not a.single_run: require(r['poison'] == int(name.split('-')[0], 16), 'poison/run identity mismatch')
        require(r['input_sha256'] == input_sha and r['expected_intermediate_opened'] is False and r['live_device_handoffs'] is True, 'runtime ingress violation')
        require([b['block'] for b in r['blocks']] == list(range(1, a.last_block + 1)), 'missing live block')
        require(r['coverage_errors'] == r['canary_errors'] == r['immutable_errors'] == 0, 'native guard failure')
        recorded = {f'block{b["block"]}.e4m3': b['output_sha256'] for b in r['blocks']}
        boundary_receipts = {}
        boundaries = [(4, 'downsample.json'), (8, 'downsample8.json')]
        if a.last_block >= 14: boundaries.append((14, 'downsample14.json'))
        if a.last_block >= 22: boundaries.append((22, 'downsample22.json'))
        for b, filename in boundaries:
            d = json.loads((run / filename).read_text())
            require(d['result'] == 'pass' and d['pid'] == r['pid'] and d['poison'] == r['poison'], 'boundary identity')
            require(d['weight_sha256'] == weight_hashes[b], 'boundary weight identity')
            require(d['expected_input_opened'] is False and d['live_projection_input'] is True, 'boundary ingress')
            require(d['coverage_errors'] == d['canary_errors'] == d['immutable_errors'] == 0, 'boundary guards')
            require(file_sha(run / f'block{b}-pooled.e4m3') == d['pooled_sha256'], 'pool artifact drift')
            recorded[f'block{b}-downsample.e4m3'] = d['output_sha256']
            boundary_receipts[str(b)] = file_sha(run / filename)
        for block in r['blocks']:
            b = block['block']
            if b >= 9:
                parent = 'block8-downsample.e4m3' if b == 9 else 'block14-downsample.e4m3' if b == 15 else f'block{b-1}.e4m3'
                require(block['input_sha256'] == recorded[parent], 'wide live predecessor identity')
        require(set(recorded) == set(targets), 'output inventory mismatch')
        for filename, expected in targets.items():
            actual = (run / filename).read_bytes()
            require(actual == expected and sha(actual) == recorded[filename], f'RTX mismatch: {name}/{filename}')
            total += len(actual)
        native_receipt = None
        if native_targets:
            nr = json.loads((run / 'native_exports.json').read_text())
            require(nr['pid'] == r['pid'] and nr['poison'] == r['poison'] and nr['result'] == 'pass', 'native export identity')
            require(nr['expected_input_opened'] is False and nr['live_device_inputs'] is True, 'native export ingress')
            require(nr['coverage_errors'] == nr['canary_errors'] == nr['immutable_errors'] == 0, 'native export guards')
            require(set(nr['outputs']) == set(native_targets), 'native export inventory')
            for filename, expected in native_targets.items():
                actual = (run / filename).read_bytes(); meta = nr['outputs'][filename]
                require(actual == expected and len(actual) == meta['bytes'] and sha(actual) == meta['sha256'], 'native RTX mismatch: '+filename)
                native_total += len(actual)
            native_receipt = file_sha(run / 'native_exports.json')
        for b in range(1, a.last_block + 1):
            paths = sorted((run / f'block{b}').glob('first-window-*'))
            require(len(paths) == (10 if b <= 4 else 11), 'missing stage samples')
            for path in paths:
                key = f'block{b}/{path.name}'; digest = file_sha(path)
                require(key not in samples or samples[key] == digest, 'sample depends on poison/process: ' + key)
                samples[key] = digest
        # The fresh RTX cutpoints are comparison-only, not runtime inputs.
        for stage in ('ffn', 'qkv'):
            directory = repo / f'build/nr_swin64/{stage}-analysis-v1'
            ref = checked_receipt(directory / 'receipt.json')
            data = (directory / 'decoded.f16').read_bytes()
            require(sha(data) == ref['tap_decoded_sha256'] and ref['mismatched_bytes'] == 0, 'RTX tap drift')
            require((run / f'block5/first-window-{stage}.f16').read_bytes() == data, 'RTX cutpoint mismatch')
        score_dir = repo / 'build/nr_swin64/score0-analysis-v1'
        score_ref = checked_receipt(score_dir / 'receipt.json'); expected = (score_dir / 'decoded.f16').read_bytes()
        require(sha(expected) == score_ref['tap_decoded_sha256'], 'score tap drift')
        score = (run / 'block5/first-window-scores.f16').read_bytes()
        require(score[:4096] + score[8192:12288] == expected, 'RTX score cutpoint mismatch')
        runs.append({'name': name, 'pid': r['pid'], 'poison': r['poison'], 'receipt_sha256': file_sha(run / 'run.json'),
                     'boundary_receipts_sha256': boundary_receipts, 'native_export_receipt_sha256': native_receipt, 'outputs_sha256': recorded})
    frozen = {str(p.relative_to(a.attempt)): file_sha(p) for p in sorted((a.attempt / 'source').rglob('*')) if p.is_file()}
    require(frozen and (a.attempt / 'source/probe.mm').is_file() and (a.attempt / 'source/prefix8.metal').is_file(), 'missing compiled source snapshot')
    sources = {}
    if not a.single_run:
        require(len(pids) == 4 and sorted(poisons) == [0x5a, 0x5a, 0xa5, 0xa5] and total == expected_total * 4, 'repeat matrix incomplete')
        for path in (a.attempt / 'source').iterdir():
            if path.is_file() and path.name not in ('probe.mm', 'prefix8.metal', 'wide128.metal', 'wide256.metal'):
                name = 'src/nr_swin/' + path.name
                require((repo / name).is_file() and file_sha(repo / name) == file_sha(path), 'compiled source drift')
                sources[name] = file_sha(path)
        for path in (a.attempt / 'source/scripts').iterdir():
            name = 'scripts/' + path.name
            require(file_sha(repo / name) == file_sha(path), 'tooling source drift')
            sources[name] = file_sha(path)
        for b in range(1, a.last_block + 1): require(file_sha(a.attempt / f'weights/block{b}.bin') == weight_hashes[b], 'staged weights drift')
    result = {'schema': 'neuralstudio.nr.encoder-prefix-verification.v1', 'result': 'pass', 'accepted': not bool(a.single_run),
              'reference_root_sha256': reference['receipt_root_sha256'], 'runtime_sha256': RUNTIME, 'graph_root_sha256': GRAPH,
              'entry_reference_root_sha256': entry_ref['receipt_root_sha256'], 'input_sha256': input_sha,
              'fresh_metal_processes': len(pids), 'full_grid_compared_bytes': total, 'mismatched_bytes': 0,
              'write_coverage_exactly_once': True, 'canaries_intact': True, 'inputs_immutable': True,
              'two_poison_repeat_proven': not bool(a.single_run), 'runs': runs, 'source_sha256': sources,
              'frozen_artifact_sha256': frozen, 'host_sha256': file_sha(a.attempt / 'probe'),
              'metallib_sha256': file_sha(a.attempt / 'prefix.metallib'), 'sample_sha256': samples,
              'claims': {'complete_blocks': list(range(1, a.last_block + 1)), 'live_block_handoffs': True, 'isolated_recorded_block0_input': True,
                         'repeat_accepted': not bool(a.single_run), 'full_model_inference': False}}
    if a.last_block >= 14:
        result['c128_metallib_sha256'] = file_sha(a.attempt / 'wide128.metallib')
    if a.last_block == 22:
        require(native_total == 34283520 * len(names), 'native export repeat coverage')
        result['c256_metallib_sha256'] = file_sha(a.attempt / 'wide256.metallib')
        result['native_export_compared_bytes'] = native_total
        result['total_compared_bytes_including_exports'] = total + native_total
        result['claims']['native_middle_and_skip_exports'] = True
    result['receipt_root_sha256'] = sha(json.dumps(result, sort_keys=True, separators=(',', ':')).encode())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(f'accepted={result["accepted"]} processes={len(pids)} compared_bytes={total} mismatched_bytes=0')
    print('receipt_root_sha256=' + result['receipt_root_sha256'])


if __name__ == '__main__': main()
