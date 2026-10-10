#!/usr/bin/env python3
"""Comparison-only Block 5 output and original model weights."""
import argparse
import hashlib
import json
from pathlib import Path
from nr_swin_reference import load_module


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('preserve reference')
    repo=Path(__file__).resolve().parents[1];snap=load_module('nr64_snap',repo/'scripts/build-rtx-oracle-manifest.py')
    oracle=json.loads((a.workspace/'outputs/dlssnr-fp8-block-oracle.json').read_text());graph=json.loads((a.workspace/'NeuralStudio/build/macos_host/dlssnr-graph-manifest.json').read_text())
    pool=bytearray((a.workspace/'work/rtx-oracle-fp8/block0.pool.bin').read_bytes())
    if sha(pool)!='1e340ab4b04af0107bba1a62e1667e304f90d62123651f019efbf1abc77a01db':raise ValueError('parent drift')
    for block in range(1,6):
        path=a.workspace/f'work/rtx-oracle-fp8/capture/pool-snapshots/snapshot-{block+2:04d}.delta';o=oracle['blocks'][block]
        if sha(path.read_bytes())!=o['delta_sha256']:raise ValueError('snapshot drift')
        pool,summary=snap.parse_snapshot_delta(path,block+2,pool)
        if summary['pool_state_sha256']!=o['pool_state_sha256']:raise ValueError('state drift')
    call=graph['launches'][6];offset=next(b['target']['byte_offset'] for b in call['arguments']['bindings'] if b['argument_offset']==8)
    raw=bytes(pool[offset:offset+8847360]);flat=bytearray()
    for wy in range(36):
        for wx in range(60):
            for tile in range(4):
                cell=(wy*2+tile%2)*120+wx*2+tile//2
                for row in range(16):
                    for c in range(0,64,4):
                        off=cell*1024+(c//32)*512+(row%8)*64+((c%16)//4)*16+(row//8)*4+((c%32)//16)*8
                        flat.extend(raw[off:off+4])
    m=json.loads((repo/'build/nr_first_inference/weights-manifest.json').read_text());dll=(repo/'models/nvidia/nvngx_dlssnr.dll').read_bytes()
    if sha(dll)!=graph['runtime_sha256']:raise ValueError('runtime drift')
    t=next(t for t in m['tensors'] if t['name']=='block5.layer0.layer');start=m['resource']['file_offset']+t['data_offset_in_resource'];weights=dll[start:start+t['data_size']]
    if sha(weights)!=t['sha256']:raise ValueError('weight drift')
    a.output.mkdir(parents=True);(a.output/'output.e4m3').write_bytes(flat);(a.output/'weight.bin').write_bytes(weights)
    r={'schema':'neuralstudio.nr.swin64-block5-reference.v1','runtime_sha256':sha(dll),'graph_root_sha256':graph['manifest_root_sha256'],
       'weight_sha256':sha(weights),'pool_state_sha256':sha(pool),'snapshot_sha256':o['delta_sha256'],'raw_output_sha256':sha(raw),
       'logical_output_sha256':sha(flat),'comparison_only':True,'full_model_inference':False,'result':'pass'}
    r['receipt_root_sha256']=sha(json.dumps(r,sort_keys=True,separators=(',',':')).encode());(a.output/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))


if __name__=='__main__':main()
