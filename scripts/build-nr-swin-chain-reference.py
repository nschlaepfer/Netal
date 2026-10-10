#!/usr/bin/env python3
"""Build comparison-only full tensors for the shifted C32 encoder prefix."""
import argparse
import hashlib
import json
from pathlib import Path
from nr_swin_reference import load_module


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('preserve prior reference')
    repo=Path(__file__).resolve().parents[1]
    snap=load_module('nr_snap',repo/'scripts/build-rtx-oracle-manifest.py')
    layout=load_module('nr_layout',repo/'scripts/build-nr-block1-buffer-reference.py')
    graph=json.loads((a.workspace/'NeuralStudio/build/macos_host/dlssnr-graph-manifest.json').read_text())
    oracle=json.loads((a.workspace/'outputs/dlssnr-fp8-block-oracle.json').read_text())
    weight_manifest=json.loads((repo/'build/nr_first_inference/weights-manifest.json').read_text())
    runtime=(repo/'models/nvidia/nvngx_dlssnr.dll').read_bytes()
    if sha(runtime)!=graph['runtime_sha256']:raise ValueError('runtime drift')
    pool=bytearray((a.workspace/'work/rtx-oracle-fp8/block0.pool.bin').read_bytes())
    if sha(pool)!='1e340ab4b04af0107bba1a62e1667e304f90d62123651f019efbf1abc77a01db':raise ValueError('parent pool drift')
    a.output.mkdir(parents=True);(a.output/'weights').mkdir();rows=[]
    for block in (1,2,3):
        observed=oracle['blocks'][block];launch=block+2
        path=a.workspace/f'work/rtx-oracle-fp8/capture/pool-snapshots/snapshot-{launch:04d}.delta'
        if sha(path.read_bytes())!=observed['delta_sha256']:raise ValueError('snapshot drift')
        pool,summary=snap.parse_snapshot_delta(path,launch,pool)
        if summary['pool_state_sha256']!=observed['pool_state_sha256']:raise ValueError('pool reconstruction drift')
        call=graph['launches'][launch-1]
        offset=next(b['target']['byte_offset'] for b in call['arguments']['bindings'] if b['argument_offset']==8)
        packed=bytes(pool[offset:offset+17694720]);flat=bytearray()
        for y in range(72):
            for x in range(120):flat.extend(layout.window(packed,x,y,True))
        tensor=next(t for t in weight_manifest['tensors'] if t['name']==f'block{block}.layer0.layer')
        start=weight_manifest['resource']['file_offset']+tensor['data_offset_in_resource']
        weight=runtime[start:start+tensor['data_size']]
        if sha(weight)!=tensor['sha256'] or len(weight)!=20672:raise ValueError('weight mismatch')
        (a.output/'weights'/f'block{block}.bin').write_bytes(weight)
        (a.output/f'block{block}.e4m3').write_bytes(flat)
        rows.append({'block':block,'kernel':call['kernel'],'grid':call['grid'],
                     'packed_output_sha256':sha(packed),'logical_output_sha256':sha(flat),
                     'weight_sha256':sha(weight),'snapshot_sha256':observed['delta_sha256'],
                     'pool_state_sha256':summary['pool_state_sha256']})
    receipt={'schema':'neuralstudio.nr.swin-c32-chain-reference.v1','runtime_sha256':graph['runtime_sha256'],
             'graph_root_sha256':graph['manifest_root_sha256'],'blocks':rows,
             'comparison_only':True,'full_model_inference':False,'result':'pass'}
    receipt['receipt_root_sha256']=sha(json.dumps(receipt,sort_keys=True,separators=(',',':')).encode())
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print('receipt_root_sha256='+receipt['receipt_root_sha256'])
    for row in rows:print(row['block'],row['logical_output_sha256'])


if __name__=='__main__':main()
