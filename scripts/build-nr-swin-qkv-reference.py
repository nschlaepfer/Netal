#!/usr/bin/env python3
"""Identify Q/K/V projections and export comparison-only RTX tensors."""
import argparse
import json
from pathlib import Path
import struct
from nr_swin_reference import load_cta,load_module


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--mlp-attempt',type=Path,required=True)
    p.add_argument('--fixtures',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('preserve previous comparison')
    repo=Path(__file__).resolve().parents[1]
    m=load_module('nr_mlp_math',repo/'scripts/analyze-nr-swin-mlp.py')
    rows=[]
    a.output.mkdir(parents=True)
    for cta_index in range(9):
        q,cta,records,pins=load_cta(repo,a.workspace,cta_index)
        decode=[q.decode_e4m3(i) for i in range(256)]
        weights=(a.fixtures/f'cta{cta_index}/inputs/weight.bin').read_bytes()
        wsha=q.sha(weights)
        if wsha!='0270fb8f406d3d0a916953ddaf6ec5d29328fe95b9d6cb408ca0c25c91108e3a':raise ValueError('weight mismatch')
        partial_path=a.mlp_attempt/f'cta{cta_index}-a5-a/partial.f16'
        partial=struct.unpack('<8192e',partial_path.read_bytes())
        input_tiles=[bytes(m.fp8(partial[6144+(tile*16+row)*32+m.dense_from_channel(k)],decode)
                           for row in range(16) for k in range(32)) for tile in range(4)]
        weight_slices={bytes(weights[8288+m.packed(k,n+j)] for k in range(32) for j in range(8)):n
                       for n in range(0,96,8)}
        if len(weight_slices)!=12:raise ValueError('ambiguous weight slices')
        expected=[None]*(64*96);coverage=[0]*(64*96);operations=[]
        for index,r in enumerate(records):
            if r['b'] not in weight_slices:continue
            if any(r['c']):raise ValueError('unexpected QKV accumulator')
            if r['a'] not in input_tiles:raise ValueError('MLP to QKV input mismatch')
            tile=input_tiles.index(r['a']);n=weight_slices[r['b']]
            d=struct.unpack('<128H',r['d'])
            computed=[]
            for row in range(16):
                for j in range(8):
                    total=sum(decode[r['a'][row*32+k]]*decode[r['b'][k*8+j]] for k in range(32))
                    computed.append(struct.unpack('<H',struct.pack('<e',total))[0])
                    offset=(tile*16+row)*96+n+j
                    expected[offset]=d[row*8+j];coverage[offset]+=1
            if computed!=list(d):raise ValueError('QKV dot mismatch')
            operations.append({'qmmaa_order':index,'tile':tile,'column':n})
        if len(operations)!=48 or set(coverage)!={1}:
            raise ValueError(f'QKV coverage {len(operations)} operations, {set(coverage)}')
        output=a.output/f'cta{cta_index}';output.mkdir()
        payload=struct.pack('<6144H',*expected)
        (output/'qkv.f16').write_bytes(payload)
        result={'cta':cta,'operation_count':len(operations),'operations':operations,
                'original_weight_sha256':wsha,'qkv_sha256':q.sha(payload),
                'mlp_native_output_sha256':q.sha(partial_path.read_bytes()),
                'source_observations':pins,'repeat_rtx_observations_equal':True,
                'qkv_input_equals_quantized_native_mlp':True,
                'comparison_only':True,'result':'pass'}
        (output/'receipt.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
        rows.append(result)
    receipt={'schema':'neuralstudio.nr.swin32-qkv-reference.v1','ctas':rows,
             'logical_shape':[64,96],'packed_weight_bytes':[8288,11360],
             'reference_diagnostic_only':True,'full_model_inference':False,'result':'pass'}
    receipt['receipt_root_sha256']=q.sha(q.canonical(receipt))
    (a.output/'receipt.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    print('receipt_root_sha256='+receipt['receipt_root_sha256'])
    print('qkv_operations=432 result=pass')


if __name__=='__main__':main()
