#!/usr/bin/env python3
"""Extract Block 4's two comparison outputs from its pinned RTX pool state."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from nr_swin_reference import load_module


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('preserve reference')
    repo=Path(__file__).resolve().parents[1]
    snap=load_module('nr_ds_snap',repo/'scripts/build-rtx-oracle-manifest.py')
    layout=load_module('nr_ds_layout',repo/'scripts/build-nr-block1-buffer-reference.py')
    oracle=json.loads((a.workspace/'outputs/dlssnr-fp8-block-oracle.json').read_text())
    pool=bytearray((a.workspace/'work/rtx-oracle-fp8/block0.pool.bin').read_bytes())
    if sha(pool)!='1e340ab4b04af0107bba1a62e1667e304f90d62123651f019efbf1abc77a01db':raise ValueError('parent pool drift')
    states=[]
    for block in range(1,5):
        path=a.workspace/f'work/rtx-oracle-fp8/capture/pool-snapshots/snapshot-{block+2:04d}.delta'
        observed=oracle['blocks'][block]
        if sha(path.read_bytes())!=observed['delta_sha256']:raise ValueError('snapshot drift')
        pool,summary=snap.parse_snapshot_delta(path,block+2,pool)
        if summary['pool_state_sha256']!=observed['pool_state_sha256']:raise ValueError('pool state drift')
        states.append({'block':block,'snapshot_sha256':observed['delta_sha256'],'pool_state_sha256':summary['pool_state_sha256']})
    graph=json.loads((a.workspace/'NeuralStudio/build/macos_host/dlssnr-graph-manifest.json').read_text())
    call=graph['launches'][5];bindings={b['argument_offset']:b['target']['byte_offset'] for b in call['arguments']['bindings']}
    core=bytes(pool[bindings[8]:bindings[8]+17694720]);down=bytes(pool[bindings[64]:bindings[64]+8847360])
    core_flat=bytearray()
    for y in range(72):
        for x in range(120):core_flat.extend(layout.window(core,x,y,True))
    down_flat=bytearray()
    for wy in range(36):
        for wx in range(60):
            for tile in range(4):
                for row in range(16):
                    x=wx*8+(tile//2)*4+row%4;y=wy*8+(tile%2)*4+row//4
                    for plane in range(4):
                        offset=(plane*288*480+y*480+x)*16
                        down_flat.extend(down[offset:offset+16])
    weights=json.loads((repo/'build/nr_first_inference/weights-manifest.json').read_text())
    runtime=(repo/'models/nvidia/nvngx_dlssnr.dll').read_bytes()
    if sha(runtime)!=graph['runtime_sha256']:raise ValueError('runtime drift')
    t=next(t for t in weights['tensors'] if t['name']=='block4.layer0.layer')
    start=weights['resource']['file_offset']+t['data_offset_in_resource'];weight=runtime[start:start+t['data_size']]
    if sha(weight)!=t['sha256']:raise ValueError('weight drift')
    a.output.mkdir(parents=True);(a.output/'weights').mkdir()
    for block in (1,2,3):shutil.copy2(repo/f'build/nr_swin_chain/reference-v1/weights/block{block}.bin',a.output/f'weights/block{block}.bin')
    (a.output/'weights/block4.bin').write_bytes(weight)
    (a.output/'block4.e4m3').write_bytes(core_flat);(a.output/'block4-downsample.e4m3').write_bytes(down_flat)
    r={'schema':'neuralstudio.nr.swin-c32-downsample-reference.v1','runtime_sha256':sha(runtime),'graph_root_sha256':graph['manifest_root_sha256'],
       'states':states,'kernel':call['kernel'],'grid':call['grid'],'weight_sha256':t['sha256'],'core_packed_sha256':sha(core),
       'core_logical_sha256':sha(core_flat),'downsample_planar_sha256':sha(down),'downsample_logical_sha256':sha(down_flat),
       'comparison_only':True,'full_model_inference':False,'result':'pass'}
    r['receipt_root_sha256']=sha(json.dumps(r,sort_keys=True,separators=(',',':')).encode())
    (a.output/'receipt.json').write_text(json.dumps(r,sort_keys=True,indent=2)+'\n')
    print(json.dumps(r,indent=2))


if __name__=='__main__':main()
