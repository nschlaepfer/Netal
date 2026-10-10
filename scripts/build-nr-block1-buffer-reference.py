#!/usr/bin/env python3
"""Export diagnostic block inputs and observed outputs from a pinned RTX state."""
import argparse
import hashlib
import json
from pathlib import Path
from nr_swin_reference import load_module


def sha(data):return hashlib.sha256(data).hexdigest()


def window(buffer,x,y,blocked=False):
    result=bytearray()
    for tile in range(4):
        for row in range(16):
            if blocked:
                cell=(y*2+(tile&1))*240+x*2+tile//2
                for c in range(0,32,4):
                    offset=cell*512+(row%8)*64+((c%16)//4)*16+(row//8)*4+(c//16)*8
                    result.extend(buffer[offset:offset+4])
                continue
            yy=y*8+(tile&1)*4+row//4;xx=x*8+(tile//2)*4+row%4
            offset=yy*15360+xx*16
            result.extend(buffer[offset:offset+16])
            result.extend(buffer[8847360+offset:8847360+offset+16])
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.output.exists():raise ValueError('preserve block reference')
    repo=Path(__file__).resolve().parents[1]
    q=load_module('nr_snapshot',repo/'scripts/build-rtx-oracle-manifest.py')
    pool_path=a.workspace/'work/rtx-oracle-fp8/block0.pool.bin';pool=bytearray(pool_path.read_bytes())
    if sha(pool)!='1e340ab4b04af0107bba1a62e1667e304f90d62123651f019efbf1abc77a01db':raise ValueError('block0 pool drift')
    input_bytes=bytes(pool[71166976:71166976+17694720])
    delta=a.workspace/'work/rtx-oracle-fp8/capture/pool-snapshots/snapshot-0003.delta'
    if sha(delta.read_bytes())!='dcadbdd09a6f134b28c0339963aee0faa7f3b2944731255c11e1512063391653':raise ValueError('block1 delta drift')
    pool,summary=q.parse_snapshot_delta(delta,3,pool)
    if summary['pool_state_sha256']!='9072a79535a32e3fe1b2392db7d97383f64ad1cbf92e88d60ea828db1fb3e833':raise ValueError('block1 state drift')
    output_bytes=bytes(pool[88861696:88861696+17694720])
    if sha(output_bytes)!='176b47ca5b7312c28717abf71a5d6b71c2333b382c09946cafc48794a52c76fe':raise ValueError('block1 output drift')
    a.output.mkdir(parents=True)
    ctas=json.loads((a.workspace/'work/block1-qmma-nine-cta-analysis/block1-qmma-nine-cta-anchors.json').read_text())['ctas']
    records=[]
    for i,cta in enumerate(ctas):
        x,y,_=cta['cta'];inputs=bytes(window(input_bytes,x,y));outputs=bytes(window(output_bytes,x,y,True))
        expected_input=(repo/f'build/nr_swin_mlp/fixtures/cta{i}/inputs/input.e4m3').read_bytes()
        if inputs!=expected_input:raise ValueError('window layout disagrees with observed tensor inputs')
        path=a.output/f'cta{i}';path.mkdir();(path/'output.e4m3').write_bytes(outputs)
        records.append({'cta':cta['cta'],'input_sha256':sha(inputs),'output_sha256':sha(outputs)})
    flat_input=bytearray();flat_output=bytearray()
    for y in range(72):
        for x in range(120):
            flat_input.extend(window(input_bytes,x,y));flat_output.extend(window(output_bytes,x,y,True))
    (a.output/'input.e4m3').write_bytes(flat_input)
    (a.output/'output.e4m3').write_bytes(flat_output)
    offsets=[(row%8)*64+((c%16)//4)*16+(row//8)*4+(c//16)*8+(c%4) for row in range(16) for c in range(32)]
    if sorted(offsets)!=list(range(512)):raise ValueError('blocked cell layout not bijective')
    receipt={'schema':'neuralstudio.nr.block1-buffer-reference.v2',
             'original_output_layout':'144x240 cells of 512 bytes in measured warp-fragment order',
             'comparison_output_layout':'CTA-major, tile-major, token-major, 32 logical E4M3 channels',
             'cell_layout_bijective':True,
             'pool_before_sha256':sha(pool_path.read_bytes()),'pool_after_sha256':sha(pool),
             'original_block1_output_sha256':sha(output_bytes),'flat_input_sha256':sha(flat_input),
             'flat_output_sha256':sha(flat_output),'window_count':8640,'rows':552960,
             'diagnostic_archived_block0_input':True,'ctas':records,'result':'pass'}
    receipt['receipt_root_sha256']=sha(json.dumps(receipt,sort_keys=True,separators=(',',':')).encode())
    (a.output/'receipt.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    print('receipt_root_sha256='+receipt['receipt_root_sha256'])


if __name__=='__main__':main()
