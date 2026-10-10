#!/usr/bin/env python3
"""Pinned comparison tensors and weights for the single-layer Swin encoder."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
from nr_swin_reference import load_module


def sha(data):return hashlib.sha256(data).hexdigest()
def flatten(raw,width,height,channels,planar=False):
    out=bytearray()
    for wy in range((height+7)//8):
        for wx in range((width+7)//8):
            for tile in range(4):
                for row in range(16):
                    x=wx*8+(tile//2)*4+row%4;y=wy*8+(tile%2)*4+row//4
                    if x>=width or y>=height:out.extend(bytes(channels));continue
                    if planar:
                        for plane in range(channels//16):
                            off=(plane*height*width+y*width+x)*16;out.extend(raw[off:off+16])
                    else:
                        cell=(y//4)*(width//4)+x//4;p=(y%4)*4+x%4
                        for c in range(0,channels,4):
                            off=cell*16*channels+(c//32)*512+(p%8)*64+((c%16)//4)*16+(p//8)*4+((c%32)//16)*8
                            out.extend(raw[off:off+4])
    return bytes(out)


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--last-block',type=int,default=8);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or not 1<=a.last_block<=22:raise ValueError('preserve reference / unsupported family')
    repo=Path(__file__).resolve().parents[1];snap=load_module('encoder_snap',repo/'scripts/build-rtx-oracle-manifest.py')
    graph=json.loads((a.workspace/'NeuralStudio/build/macos_host/dlssnr-graph-manifest.json').read_text());oracle=json.loads((a.workspace/'outputs/dlssnr-fp8-block-oracle.json').read_text())
    weights=json.loads((repo/'build/nr_first_inference/weights-manifest.json').read_text());dll=(repo/'models/nvidia/nvngx_dlssnr.dll').read_bytes()
    if sha(dll)!=graph['runtime_sha256']:raise ValueError('runtime drift')
    pool=bytearray((a.workspace/'work/rtx-oracle-fp8/block0.pool.bin').read_bytes())
    if sha(pool)!='1e340ab4b04af0107bba1a62e1667e304f90d62123651f019efbf1abc77a01db':raise ValueError('parent pool drift')
    a.output.mkdir(parents=True);(a.output/'weights').mkdir();records=[]
    for block in range(1,a.last_block+1):
        delta=a.workspace/f'work/rtx-oracle-fp8/capture/pool-snapshots/snapshot-{block+2:04d}.delta';o=oracle['blocks'][block]
        if sha(delta.read_bytes())!=o['delta_sha256']:raise ValueError('snapshot drift')
        pool,summary=snap.parse_snapshot_delta(delta,block+2,pool)
        if summary['pool_state_sha256']!=o['pool_state_sha256']:raise ValueError('state drift')
        call=graph['launches'][block+1];channels=int(re.search(r'_swin_\dh_(\d+)_',call['kernel']).group(1))
        constants=bytes.fromhex(call['arguments']['constant_bytes_hex']);height,width,x,y=struct.unpack_from('<4i',constants,24 if channels==32 else 32)
        if width%4 or height%4:raise ValueError('unsupported cell padding')
        bindings={b['argument_offset']:b['target']['byte_offset'] for b in call['arguments']['bindings']}
        raw=bytes(pool[bindings[8]:bindings[8]+width*height*channels]);flat=flatten(raw,width,height,channels)
        (a.output/f'block{block}.e4m3').write_bytes(flat)
        t=next(t for t in weights['tensors'] if t['name']==f'block{block}.layer0.layer');start=weights['resource']['file_offset']+t['data_offset_in_resource'];w=dll[start:start+t['data_size']]
        if sha(w)!=t['sha256']:raise ValueError('weight drift')
        (a.output/f'weights/block{block}.bin').write_bytes(w)
        row={'block':block,'width':width,'height':height,'channels':channels,'origin':[x,y],'grid':call['grid'],'kernel':call['kernel'],
             'weight_sha256':sha(w),'snapshot_sha256':o['delta_sha256'],'pool_state_sha256':o['pool_state_sha256'],
             'raw_output_sha256':sha(raw),'outputs':{f'block{block}.e4m3':sha(flat)}}
        if '_ds_' in call['kernel']:
            arg=64 if channels==32 else 72;dh,dw=struct.unpack_from('<2i',constants,arg+8)
            if (dh,dw)!=(height//2,width//2):raise ValueError('downsample shape differs from selected assumption')
            raw=bytes(pool[bindings[arg]:bindings[arg]+dw*dh*channels*2]);flat=flatten(raw,dw,dh,channels*2,True)
            (a.output/f'block{block}-downsample.e4m3').write_bytes(flat);row['downsample_raw_sha256']=sha(raw);row['outputs'][f'block{block}-downsample.e4m3']=sha(flat)
        records.append(row);print(block,row['outputs'],flush=True)
    result={'schema':'neuralstudio.nr.encoder-reference.v1','runtime_sha256':sha(dll),'graph_root_sha256':graph['manifest_root_sha256'],'blocks':records,'comparison_only':True,'full_model_inference':False,'result':'pass'}
    result['receipt_root_sha256']=sha(json.dumps(result,sort_keys=True,separators=(',',':')).encode());(a.output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(result['receipt_root_sha256'])


if __name__=='__main__':main()
