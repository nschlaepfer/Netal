#!/usr/bin/env python3
"""Counterexample to skipping normalize/denormalize rounding in a centre lemma.

Uses the peer runtime's RCP-ROM/multiply model, not a new RTX hardware result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def bits(x):return struct.unpack('<I',struct.pack('<f',x))[0]
def value(x):return struct.unpack('<f',struct.pack('<I',x))[0]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--peer-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    rompath=a.peer_root/'build/nr_ptx/tables/sm120-rcp-correction-rom.bin';rom=rompath.read_bytes();rt=a.peer_root/'src/nr_ptx/nr_ptx_rt.metal';text=rt.read_text()
    assert 'inline float nr_div_approx(float a, float b, device const uchar* rom) { return a * nr_rcp_rom(b, rom); }' in text
    rows=[]
    for width in (1920,1080):
        mant=bits(width)&0x7fffff;code=(rom[mant>>2]>>(2*(mant&3)))&3;rcp=value(bits(1/width)+(-1 if code==1 else 1 if code==2 else 0));bad=[]
        for x in range(width):
            centre=f32(x+.5);uv=f32(centre*rcp);returned=f32(uv*width)
            if centre!=returned:bad.append({'pixel':x,'ideal_texel_centre_bits':f'{bits(centre):08x}','normalized_bits':f'{bits(uv):08x}','returned_texel_bits':f'{bits(returned):08x}','error':returned-centre})
        rows.append({'dimension':width,'reciprocal_bits':f'{bits(rcp):08x}','rom_code':code,'noncentre_positions':len(bad),'below_centre':sum(v['error']<0 for v in bad),'above_centre':sum(v['error']>0 for v in bad),'counterexamples':bad})
    assert [r['noncentre_positions'] for r in rows]==[1168,229]
    r={'schema':'neuralstudio.nr.zero-motion-coordinate-review.v1','result':'centre_shortcut_refuted_in_current_runtime_model','axes':rows,'runtime_sha256':sha(rt),'rom_sha256':sha(rompath),'cuda_executed':False,'scope':'The deployed div/rcp model followed by f32 multiplication back to texel coordinates does not recover exact centres everywhere. Zero-motion FMA leaves normalized coordinates unchanged. Peer centre-only test omits this round trip. This does not prove the actual CUDA sampler quantisation rule or that the resulting final half pixels differ; use exact coordinates in tests and retain hardware validation.','verifier_sha256':sha(Path(__file__))}
    r['receipt_root_sha256']=canonical(r);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'root':r['receipt_root_sha256'],'axes':[{k:v for k,v in x.items() if k!='counterexamples'} for x in rows]},indent=2))
if __name__=='__main__':main()
