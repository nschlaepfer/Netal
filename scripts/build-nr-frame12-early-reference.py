#!/usr/bin/env python3
"""CPU-only early encoder extraction; comparison payloads never enter inference."""
import argparse
import importlib.util
import json
from pathlib import Path
import struct


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_layout(packed):
    # Same bijective blocked-cell layout used by the accepted C32 reference.
    assert len(packed) == 17694720
    output = bytearray()
    for y in range(72):
        for x in range(120):
            for tile in range(4):
                cell = (y * 2 + (tile & 1)) * 240 + x * 2 + tile // 2
                for row in range(16):
                    for c in range(0, 32, 4):
                        offset = cell * 512 + (row % 8) * 64 + ((c % 16) // 4) * 16 + (row // 8) * 4 + (c // 16) * 8
                        output.extend(packed[offset:offset + 4])
    assert len(output) == len(packed)
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--job-root', type=Path, required=True)
    p.add_argument('--label', default='continuous-a5-a')
    p.add_argument('--parent-reference', type=Path, required=True)
    a = p.parse_args()
    job = a.job_root.resolve()
    assert job.drive.upper() == 'D:' and job.parent.name == 'CodexLab'
    assert job.name == 'nr-color-reset-early12-20260906-v1'
    audit = load('sixteen', job / 'build-nr-sixteen-reference.py')
    parent = audit.read(a.parent_reference)
    body = dict(parent)
    root = body.pop('receipt_root_sha256')
    assert audit.canonical(body) == root == 'de1a27a1dc001362ee6ee0b838347c47a49b5c678697b3fb31cee5113388029e'
    out = job / 'early-reference-v1'
    assert not out.exists(), 'preserve previous extraction'
    out.mkdir()
    capture = job / a.label
    events = [json.loads(s) for s in (capture / 'rtx-cuda-trace.jsonl').read_text(encoding='utf-8-sig').splitlines()]
    alloc = next(e for e in events if e.get('call') == 'cuMemAlloc' and e['allocation_id'] == 3)
    assert alloc['bytes'] == 212519936
    pointer = int(str(alloc['pointer']), 16)
    launches = {e['launch_id']: e for e in events if e.get('call') == 'cuLaunchKernel'}
    offsets = [88861696, 106556416, 124251136, 141945856]
    blocks = []
    def observe(launch, state, info):
        if launch not in (1875, 1876, 1877, 1878):
            return
        block = launch - 1874
        call = launches[launch]
        args = (capture / 'cuda-dumps' / call['argument_dump']).read_bytes()
        offset = struct.unpack_from('<Q', args, 8)[0] - pointer
        assert offset == offsets[block - 1], 'live output binding differs from selected original graph'
        input_offset = struct.unpack_from('<Q', args, 0)[0] - pointer
        assert input_offset == ([71166976] + offsets)[block - 1]
        packed = bytes(memoryview(state)[offset:offset + 17694720])
        target = out / f'block{block}.e4m3'
        target.write_bytes(canonical_layout(packed))
        blocks.append({'block': block, 'launch': launch, 'input_pool_offset': input_offset,
                       'output_pool_offset': offset, 'kernel': call['name'], 'snapshot': info,
                       'packed_sha256': audit.hashlib.sha256(packed).hexdigest(),
                       'canonical_sha256': audit.sha(target), 'bytes': target.stat().st_size})
    r = audit.build(job, a.label, job / 'build-rtx-oracle-manifest.py',
                    source_root='51be564882e6af7e2fca84b198ba46d1cf1603fa568db12382c5bc7968092863',
                    snapshot_ids=sorted([156*i+j for i in range(16) for j in (2, 24)] + [1875, 1876, 1877, 1878]),
                    snapshot_observer=observe)
    assert r['outputs'] == parent['outputs'] and r['history_sha256'] == parent['history_sha256']
    assert [(b['launch'], b['ports']) for b in r['boundaries']] == [(b['launch'], b['ports']) for b in parent['boundaries']]
    assert [b['block'] for b in blocks] == [1, 2, 3, 4]
    r.update(schema='neuralstudio.nr.frame12-early-reference.v1', early_blocks=blocks,
             parent_reference_root=root, all_sixteen_outputs_and_history_unchanged=True,
             all_original_boundary_ports_unchanged=True, wrapper_sha256=audit.sha(Path(__file__)),
             observation_scope='one fresh-process early-boundary diagnostic, not repeated boundary acceptance')
    r['receipt_root_sha256'] = audit.canonical(r)
    receipt = out / 'reference.json'
    receipt.write_text(json.dumps(r, indent=2) + '\n')
    print(json.dumps({'reference_root': r['receipt_root_sha256'], 'blocks': blocks}, indent=2), flush=True)
    pack = load('pack', job / 'pack-nr-reset-retrieval.py')
    transport = pack.pack({p.name: p for p in out.iterdir()}, job / 'transport-v1', 'early-reference-v1')
    print(json.dumps({'transport_root': transport['receipt_root_sha256'], 'parts': len(transport['parts'])}), flush=True)


if __name__ == '__main__':
    main()
