#!/usr/bin/env python3
"""Pin selected Block0 PTX and add an observation-only shared feature tap."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();root=Path(__file__).resolve().parents[1]
    graph=root.parent/'NeuralStudio/build/macos_host/dlssnr-graph-manifest.json';g=json.loads(graph.read_text());launch=g['launches'][1]
    assert launch['kernel']=='cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8'
    assert launch['grid']==[240,144,1] and launch['block']==[32,1,1]
    manifest=json.loads((root/'build/nr_first_inference/weights-manifest.json').read_text())
    runtime=root/'models/nvidia/nvngx_dlssnr.dll';assert sha(runtime)==manifest['runtime']['sha256']
    tensor=next(x for x in manifest['tensors'] if x['name']=='block0.layer0.layer')
    with runtime.open('rb') as f:
        f.seek(manifest['resource']['file_offset']+tensor['data_offset_in_resource']);weight=f.read(tensor['data_size'])
    assert hashlib.sha256(weight).hexdigest()==tensor['sha256']
    source=root/'build/nr_pre/ptx-v1/block0.ptx';assert sha(source)=='de51a63233189fee3eb4fdfe85cd99eb31b9a6042cfbd08fd9464fe5cba8d8bc'
    text=source.read_text();shared=re.search(r'\.shared \.align 16 \.b8 (\w+)\[2048\];',text).group(1)
    # New temporaries, no reassignment of any original register. Snapshot after
    # the original barrier, and before any consumer can overwrite shared data.
    tap=['{','.reg .b32 t<8>;','.reg .b64 d<4>;','ld.param.b64 d0, [%rd16+264];','ld.param.b64 d1, [%rd16+272];',
         'mov.u32 t0, %ctaid.y;','mov.u32 t1, %ctaid.x;','mad.lo.u32 t0, t0, 240, t1;',
         'mov.u32 t1, %tid.x;','mul.lo.u32 t2, t1, 4;',f'mov.u32 t3, {shared};','add.u32 t3, t3, t2;',
         'mul.lo.u32 t0, t0, 2048;','add.u32 t0, t0, t2;','cvt.u64.u32 d2, t0;','add.u64 d0, d0, d2;','add.u64 d1, d1, d2;']
    for j in range(16):tap += [f'ld.shared.b32 t4, [t3+{j*128}];',f'st.global.b32 [d0+{j*128}], t4;',f'add.u64 d3, d1, {j*128};','atom.global.add.u32 t5, [d3], 1;']
    tap += ['bar.sync 0;','}']
    assert text.count('param_0[264]')==1
    tapped=text.replace('param_0[264]','param_0[280]').replace('bar.sync 0;','bar.sync 0;\n'+'\n'.join(tap),1)
    a.output.mkdir(parents=True,exist_ok=False);shutil.copyfile(source,a.output/'original.ptx')
    (a.output/'feature-tap.ptx').write_text(tapped);(a.output/'weights.bin').write_bytes(weight)
    (a.output/'params.bin').write_bytes(bytes.fromhex(launch['arguments']['constant_bytes_hex']))
    image=root.parent/'work/rtx-oracle-fp8/capture/input-color.rgba16le'
    assert sha(image)=='81ec1de5d1e76e6b55c4fd987b3732180c9720919fbd760dd5b4e8aa8cdd0342'
    shutil.copyfile(image,a.output/'input.rgba16le');shutil.copyfile(root/'src/nr_pre/pre_oracle.cu',a.output/'pre_oracle.cu')
    shutil.copyfile(root/'scripts/run-nr-pre-oracle.ps1',a.output/'run.ps1')
    r={'schema':'neuralstudio.nr.pre-oracle-inputs.v1','runtime_sha256':manifest['runtime']['sha256'],
       'graph_sha256':sha(graph),'selected_launch':2,'files':{x.name:sha(x) for x in sorted(a.output.iterdir())},
       'tap':'raw 2048-byte shared feature tile after original first barrier; continues complete original kernel',
       'expected_answers_staged':False}
    r['receipt_root_sha256']=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (a.output/'inputs.json').write_text(json.dumps(r,indent=2)+'\n');print(r['receipt_root_sha256'])

if __name__=='__main__':main()
