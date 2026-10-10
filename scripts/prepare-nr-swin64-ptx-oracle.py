#!/usr/bin/env python3
"""Freeze an isolated selected-PTX replay and observational FFN register tap."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import re


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',choices=['ffn','qkv','score0'],default='ffn');a=p.parse_args()
    if a.output.exists():raise ValueError('preserve job')
    repo=Path(__file__).resolve().parents[1];a.output.mkdir(parents=True)
    source=repo/'build/nr_swin64/ptx-v1/block5.ptx';ptx=source.read_text()
    if sha(source.read_bytes())!='214bfd8e6547a56e92ba0c8111d371d7d129722b3c780250c8ea471a81a89244':raise ValueError('selected PTX drift')
    if 'param_0+24]' in ptx:raise ValueError('trace argument conflicts with model argument')
    marker='@%p17 bra $L__BB17_43;\n'
    registers=[f'%r{4635+i}' for i in range(32)]
    if a.stage=='qkv':
        marker='ld.global.b32 %r2117, [%rd248+57504];\n'
        section=ptx[ptx.index('add.s64 %rd197, %rd224, 28832;'):ptx.index(marker)]
        mmas=re.findall(r'mma\.sync[^;]*;',section)
        if len(mmas)!=96:raise ValueError('QKV instruction schedule drift')
        registers=[r for mma in mmas[48:] for r in re.findall(r'%r\d+',mma.split('}')[0])]
        if len(set(registers))!=96:raise ValueError('QKV result register ambiguity')
    if a.stage=='score0':
        section=ptx[ptx.index('add.s64 %rd209, %rd251, 41120;'):]
        mmas=re.findall(r'mma\.sync[^;]*;',section)[:16]
        registers=[r for mma in mmas for r in re.findall(r'%r\d+',mma.split('}')[0])]
        if len(set(registers))!=32:raise ValueError('score result register ambiguity')
        marker=mmas[-1]
    tap_bytes=len(registers)*2*32*4
    if ptx.count(marker)!=1:raise ValueError('FFN boundary ambiguous')
    declarations='\n.reg .u32 %nr_lane, %nr_warp, %nr_x, %nr_y, %nr_index, %nr_old;\n.reg .u64 %nr_pointer, %nr_offset, %nr_address;\n.reg .pred %nr_skip;\n'
    tapped=ptx.replace('.reg .b64 %rd<389>;', '.reg .b64 %rd<389>;'+declarations)
    instructions=['mov.u32 %nr_x, %ctaid.x;','mov.u32 %nr_y, %ctaid.y;','or.b32 %nr_index, %nr_x, %nr_y;',
                  'setp.ne.u32 %nr_skip, %nr_index, 0;','@%nr_skip bra NR_FFN_TAP_DONE;',
                  'ld.param.u64 %nr_pointer, [cc_tinlayout_fused_swin_2h_64_2_inpview_tilesync_fp8_param_0+24];',
                  'mov.u32 %nr_lane, %laneid;','mov.u32 %nr_warp, %tid.y;',
                  f'mad.lo.u32 %nr_index, %nr_warp, {len(registers)*32}, %nr_lane;','mul.wide.u32 %nr_offset, %nr_index, 4;']
    for i,register in enumerate(registers):
        instructions += ['add.u64 %nr_address, %nr_pointer, %nr_offset;',f'add.u64 %nr_address, %nr_address, {i*128};',
                         f'st.global.u32 [%nr_address], {register};',f'add.u64 %nr_address, %nr_address, {tap_bytes};',
                         'atom.global.add.u32 %nr_old, [%nr_address], 1;']
    instructions.append('NR_FFN_TAP_DONE:')
    tapped=tapped.replace(marker,marker+'\n'+'\n'.join(instructions)+'\n')
    (a.output/'original.ptx').write_text(ptx);(a.output/f'{a.stage}-tap.ptx').write_text(tapped)
    flat=(repo/'build/nr_swin_downsample/reference-v1/block4-downsample.e4m3').read_bytes();planar=bytearray(len(flat))
    for wy in range(36):
        for wx in range(60):
            for tile in range(4):
                for row in range(16):
                    x=wx*8+(tile//2)*4+row%4;y=wy*8+(tile%2)*4+row//4;start=(((wy*60+wx)*4+tile)*16+row)*64
                    for plane in range(4):
                        offset=(plane*288*480+y*480+x)*16;planar[offset:offset+16]=flat[start+plane*16:start+plane*16+16]
    if sha(planar)!='8479e73ec995a2376d9345515d713b20c5d6088f6f858c7aefd2b0978b4f939b':raise ValueError('diagnostic entry conversion drift')
    (a.output/'input.planar').write_bytes(planar)
    shutil.copy2(repo/'build/nr_swin64/reference-v1/weight.bin',a.output/'weight.bin')
    shutil.copy2(repo/'src/nr_swin/ptx_oracle.cu',a.output/'ptx_oracle.cu')
    shutil.copy2(repo/'scripts/run-nr-swin64-ptx.ps1',a.output/'run.ps1')
    r={'schema':'neuralstudio.nr.swin64-ptx-job-inputs.v1','runtime_sha256':'e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e',
       'selected_entry_sha256':sha(source.read_bytes()),'purpose':'Isolated recorded-input Block 5 diagnostic, not whole-model execution',
       'tap':'Observational stores at a named boundary; only CTA 0,0, both warps; original math unmodified',
       'tap_name':a.stage+'-tap','tap_bytes':tap_bytes,'registers':registers,'stage':a.stage,
       'expected_bytes_supplied':False,'files':{x.name:sha(x.read_bytes()) for x in sorted(a.output.iterdir()) if x.is_file()}}
    r['receipt_root_sha256']=sha(json.dumps(r,sort_keys=True,separators=(',',':')).encode());(a.output/'inputs.json').write_text(json.dumps(r,indent=2)+'\n');print(r['receipt_root_sha256'])


if __name__=='__main__':main()
