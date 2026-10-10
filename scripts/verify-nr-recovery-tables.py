#!/usr/bin/env python3
"""Produce a read-only handoff of surviving NR scalar arithmetic tables.

No GPU execution, table modification, weights or recorded model tensors enter
this verification. Historical scalar observations are not new model parity.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def canonical(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def half_table(directory, source, operation, expected):
    job = read(directory / 'receipt.json')
    need(job['result'] == 'pass' and job['gpu'] == 'NVIDIA GeForce RTX 5090, 616.64', 'historical oracle identity')
    need(sha(source) == job['source_sha256'], 'oracle source identity')
    names = {f'{label}.{suffix}' for label in ('a5-a', 'a5-b', '5a-a', '5a-b') for suffix in ('bin', 'json', 'log')}
    need(len(job['artifacts']) == 12 and {a['name'] for a in job['artifacts']} == names, 'complete unique artifact inventory')
    for artifact in job['artifacts']:
        path = directory / 'output' / artifact['name']
        need(path.stat().st_size == artifact['bytes'] and sha(path) == artifact['sha256'], 'retained artifact bytes')
    pids = set()
    for label, poison in [('a5-a', 165), ('a5-b', 165), ('5a-a', 90), ('5a-b', 90)]:
        record = read(directory / 'output' / (label + '.json'))
        need(record['schema'] == f'neuralstudio.nr.{operation}-half-rtx.v1', 'operation schema')
        need(record['result'] == 'pass' and record['count'] == 31743 and record['errors'] == 0, 'complete guarded scalar observation')
        need(record['poison'] == poison and record['pid'] > 0 and record['pid'] not in pids, 'fresh process and poison')
        need(record['device'] == 'NVIDIA GeForce RTX 5090' and record['expected_input_opened'] is False, 'scalar-only oracle')
        instruction = 'rsqrt' if operation == 'rsq' else 'rcp'
        need(record['instruction'] == instruction + '.approx.ftz.f32 + cvt.rn.f16.f32', 'exact instruction contract')
        need(sha(directory / 'output' / (label + '.bin')) == expected, 'pinned repeated table bytes')
        pids.add(record['pid'])
    path = directory / 'output/a5-a.bin'
    return {'path': str(path), 'sha256': expected, 'bytes': path.stat().st_size,
            'job_receipt_sha256': sha(directory / 'receipt.json'),
            'source': str(source), 'source_sha256': sha(source), 'fresh_historical_pids': sorted(pids),
            'domain': 'positive finite binary16 bit patterns 1..0x7bff; little-endian uint16 result at index bits-1',
            'semantics': 'RTX approx.ftz.f32 followed by round-to-nearest binary16; NOT general FP32 SFU output',
            'limits': 'Guards/coverage summarized by historical errors=0; no new GPU execution or per-process executable rehash.'}


def audit():
    # Historical final half-result hashes, not frame/pixel/model-output keys.
    tables = {
        'rtx-rsqrt-half.bin': half_table(ROOT / 'build/nr_rsq_half/attempt-v2', ROOT / 'src/nr_swin/rsq_half_oracle.cu', 'rsq', '5d30a7dbcb4333b769c53e5404b5bd1737e9761403021cf71a4f415a3d96aad7'),
        'rtx-rcp-half.bin': half_table(ROOT / 'build/nr_rcp_half/rtx-v1', ROOT / 'build/nr_rcp_half/source-v1/rsq_half_oracle.cu', 'rcp', 'dfb76ea177f91a47247b7096829e2b40ce05cf0833749066574a97c56eb568f4')}
    spec = importlib.util.spec_from_file_location('ex2', ROOT / 'scripts/verify-nr-ex2-half-domain.py')
    ex2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex2)
    need(__debug__, 'EX2 historical verifier requires assertions; do not use Python -O')
    data, report = ex2.audit([ROOT / 'build/nr_ex2_half' / n for n in ('a5-a', 'a5-b', '5a-a', '5a-b')])
    path = ROOT / 'build/nr_ex2_half/table-v1/ex2-scaled-half.u32x2'
    need(data == path.read_bytes() and sha(path) == 'f9ae941e60868e72489db582fe725bdb401e033fce8694ecd35dd0c3329ba3cf', 'EX2 retained bytes versus all scalar observations')
    tables['ex2-scaled-half.u32x2'] = {'path': str(path), 'bytes': len(data), 'sha256': sha(path),
        'domain_rows': report['domain_rows'], 'verified_finite_rows': report['finite_argument_rows_verified'],
        'integration_contract': report['integration_contract'], 'limits': report['nonfinite_rows']}
    bundle = ROOT / 'build/nr_encoder_telemetry/zero-v3'
    build = read(bundle / 'build.json')
    root = build.pop('receipt_root_sha256')
    need(canonical(build) == root == 'e4a60e4442cf88efb5cf09bfa631da81268ea86a39fce96c877ad030dacdc6aa', 'frozen encoder parent seal')
    path = bundle / 'assets/temporal-rcp-rom.bin'
    expected = '7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750'
    need(build['assets']['temporal-rcp-rom.bin'] == sha(path) == expected, 'accepted encoder reciprocal ROM identity')
    tables['sm120-rcp-correction-rom.bin'] = {'path': str(path), 'sha256': expected,
        'bytes': path.stat().st_size, 'encoder_parent_root': root,
        'scope': 'byte recovery of the existing reciprocal-ROM primitive only; keep original index/domain/fallback logic'}
    # Read-only lookup in FG's surviving arithmetic archive. These exact files
    # were already used by NR; this is not a new FG backend or model port.
    parent = ROOT / 'build/nr_resident_review/live-v17-black3-v2.json'
    ledger = read(parent)
    ledger_root = ledger.pop('receipt_root_sha256')
    need(canonical(ledger) == ledger_root == '5bd81e7b5094114f3a1efd55a1a952f9abfaea1d3e2aaacda46e1a3326d92a50', 'historical NR artifact-map seal')
    legacy = ROOT.parent / 'NeuralStudio-fg-research/build/fg_math_ex2/compact/attempt-v4'
    for name, expected in [('unit', '60c2ae1a0869ce6f10ba093a33def68ecdbf4ba0d789cd186289586b63b9f234'),
                           ('fraction', '76bacb1adf8e53accdd6a40a64940c073f001e091232e9119b155f2ab25f1a13')]:
        filename = name + '-corrections.bin'
        accepted = [h for p, h in ledger['artifacts'].items() if p.endswith('/tables/' + filename)]
        need(accepted == [expected], 'historical NR correction-table pin')
        path = legacy / 'packed-a' / filename
        need(sha(path) == sha(legacy / 'packed-b' / filename) == expected, 'both surviving packed copies')
        record = read(legacy / 'packed-a' / (name + '-receipt.json'))
        need(record['packed_sha256'] == expected and record['packed_bytes'] == path.stat().st_size and record['reconstruction_mismatches'] == 0, 'historical packing receipt')
        tables[filename] = {'path': str(path), 'sha256': expected, 'bytes': path.stat().st_size,
            'historical_nr_ledger_root': ledger_root,
            'scope': 'existing packed scalar EX2 corrections; preserve original decoder, domain, predictor and fallback semantics',
            'limits': 'Read-only recovery of exact NR-pinned bytes; original FG oracle census not rerun.'}
    return {'schema': 'neuralstudio.nr.recovery-tables.v1', 'result': 'retained_scalar_tables_verified',
            'tables': tables, 'model_executed': False, 'runtime_changed': False, 'full_model_parity': False,
            'reference_tensor_inputs': False, 'verifier_sha256': sha(Path(__file__)),
            'limits': ['Handoff paths only; no new tables produced and no peer files changed.',
                       'Auxiliary fast-exp2 dumps were not in the historical NR loaded-artifact ledger; do not infer new runtime dependencies.',
                       'Binding these tables into a reconstructed body still requires fresh complete-model validation.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    need(not args.out.exists(), 'refusing to overwrite receipt')
    result = audit()
    result['receipt_root_sha256'] = canonical(result)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
